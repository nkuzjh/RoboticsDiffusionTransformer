"""Focused CPU checks for aligned RDT diffusion and role adaptation."""

from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import torch

from models.csgo_adaptation import RoleLoRALinear, build_role_adaptation
from models.csgo_rdt import CSGORDTRunner


def tiny_runner(policy: str = "native_full") -> CSGORDTRunner:
    return CSGORDTRunner(
        config={
            "lang_adaptor": "mlp2x_gelu",
            "img_adaptor": "mlp2x_gelu",
            "state_adaptor": "mlp3x_gelu",
            "lang_token_dim": 8,
            "img_token_dim": 6,
            "state_token_dim": 128,
            "rdt": {"hidden_size": 32, "depth": 2, "num_heads": 4},
            "noise_scheduler": {
                "num_train_timesteps": 10, "num_inference_timesteps": 2,
                "beta_schedule": "squaredcos_cap_v2", "prediction_type": "sample",
                "clip_sample": False,
            },
        },
        lang_token_dim=8,
        img_token_dim=6,
        state_token_dim=128,
        max_lang_cond_len=4,
        img_cond_len=2,
        dtype=torch.float32,
        diffusion_channel_policy=policy,
    )


def inputs():
    return (
        torch.randn(2, 4, 8), torch.ones(2, 4, dtype=torch.bool),
        torch.randn(2, 2, 6), torch.zeros(2, 1, 128),
        torch.tensor([[[1., 2., 3., 4., 5.]], [[2., 3., 4., 5., 6.]]]),
        torch.tensor([[[1., 1., 1., 1., 1.]], [[1., 1., 1., 1., 1.]]]),
        torch.ones(2),
    )


class AlignedRDTTests(unittest.TestCase):
    def test_full_width_noise_loss_and_final_only_mask(self):
        model = tiny_runner()
        call = inputs()
        with patch.object(model.noise_scheduler, "add_noise", wraps=model.noise_scheduler.add_noise) as add_noise:
            loss = model.compute_loss(*call)
        action, noise, _ = add_noise.call_args.args
        self.assertEqual(action.shape, (2, 1, 128))
        self.assertTrue(torch.all(action[..., 5:] == 0))
        self.assertTrue(torch.any(noise[..., 5:] != 0))
        self.assertTrue(loss.isfinite())
        # Initial final-layer prediction is zero, so the full 128D MSE is exact.
        self.assertAlmostEqual(loss.item(), call[4].square().sum().item() / (2 * 128), places=5)

        full_sample = []
        original_sample = model.conditional_sample

        def capture_sample(*args):
            value = original_sample(*args)
            full_sample.append(value)
            return value

        with patch.object(model.noise_scheduler_sample, "step", wraps=model.noise_scheduler_sample.step) as step, \
                patch.object(model, "conditional_sample", side_effect=capture_sample):
            result = model.predict_action(call[0], call[1], call[2], call[3], call[5], call[6])
        self.assertEqual(result.shape, (2, 1, 5))
        self.assertTrue(torch.any(step.call_args_list[0].args[2][..., 5:] != 0))
        self.assertTrue(torch.all(full_sample[0][..., 5:] == 0))

    def test_adaptation_groups_gradient_and_checkpoint_roundtrip(self):
        model = tiny_runner()
        adaptation = build_role_adaptation(model, {"training": {"adaptation_mode": "role_lora"}})
        self.assertEqual(len(adaptation["optimizer_groups"]), 7)
        self.assertEqual(adaptation["audit"]["lora_layers"], 14)
        self.assertTrue(isinstance(model.model.blocks[0].attn.qkv, RoleLoRALinear))
        self.assertTrue(all(not p.requires_grad for p in model.model.blocks[0].attn.qkv.base.parameters()))
        self.assertTrue(all(p.dtype == torch.float32 for group in adaptation["optimizer_groups"] for p in group["params"]))
        loss = model.compute_loss(*inputs())
        loss.backward()
        self.assertIsNotNone(model.model.final_layer.ffn_final.fc2.weight.grad)
        self.assertIsNotNone(model.model.blocks[0].attn.qkv.lora_B.weight.grad)

        with tempfile.TemporaryDirectory() as temp:
            model.save_pretrained(temp)
            loaded = CSGORDTRunner.from_pretrained(temp, dtype=torch.float32)
            self.assertEqual(loaded.diffusion_channel_policy, "native_full")
            self.assertEqual(loaded.role_adaptation["mode"], "role_lora")
            self.assertTrue(isinstance(loaded.model.blocks[0].attn.qkv, RoleLoRALinear))
            self.assertEqual(set(model.state_dict()), set(loaded.state_dict()))
            self.assertTrue(torch.equal(
                loaded.model.blocks[0].attn.qkv.lora_A.weight,
                model.model.blocks[0].attn.qkv.lora_A.weight,
            ))
            self.assertEqual(loaded._csgo_load_report["missing_keys"], [])

            from safetensors.torch import save_file

            safe_dir = Path(temp) / "accelerator_format"
            safe_dir.mkdir()
            with (safe_dir / "config.json").open("w", encoding="utf-8") as stream:
                json.dump(model._hub_mixin_config, stream)
            save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()},
                      str(safe_dir / "model.safetensors"))
            safe_loaded = CSGORDTRunner.from_pretrained(safe_dir, dtype=torch.float32)
            self.assertEqual(safe_loaded._csgo_load_report["missing_keys"], [])

    def test_legacy_default_and_invalid_padding(self):
        self.assertEqual(tiny_runner("legacy_valid5").diffusion_channel_policy, "legacy_valid5")
        model = tiny_runner()
        call = list(inputs())
        padded = torch.zeros(2, 1, 128)
        padded[..., :5] = call[4]
        padded[..., 6] = 1
        call[4] = padded
        with self.assertRaisesRegex(ValueError, "padded CSGO target"):
            model.compute_loss(*call)

    def test_bf16_base_fp32_trainables_support_train_and_predict(self):
        model = tiny_runner().to(dtype=torch.bfloat16)
        adaptation = build_role_adaptation(model, {"adaptation_mode": "role_lora"})
        self.assertEqual(model.model.blocks[0].attn.qkv.base.weight.dtype, torch.bfloat16)
        self.assertTrue(all(p.dtype == torch.float32 for group in adaptation["optimizer_groups"] for p in group["params"]))
        loss = model.compute_loss(*inputs())
        loss.backward()
        self.assertTrue(loss.isfinite())
        self.assertIsNotNone(model.model.blocks[0].attn.qkv.lora_B.weight.grad)
        call = inputs()
        with torch.no_grad():
            action = model.predict_action(call[0], call[1], call[2], call[3], call[5], call[6])
        self.assertEqual(action.shape, (2, 1, 5))
        self.assertTrue(torch.isfinite(action).all())

    def test_incomplete_or_mismatched_lora_checkpoint_is_rejected(self):
        model = tiny_runner()
        build_role_adaptation(model, {"adaptation_mode": "role_lora"})
        with tempfile.TemporaryDirectory() as temp:
            model.save_pretrained(temp)
            state_path = Path(temp) / "pytorch_model.bin"
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            state.pop("model.blocks.0.attn.qkv.lora_A.weight")
            torch.save(state, state_path)
            with self.assertRaisesRegex(RuntimeError, "structural validation"):
                CSGORDTRunner.from_pretrained(temp, dtype=torch.float32)

            config_path = Path(temp) / "config.json"
            config = json.loads(config_path.read_text())
            config["role_adaptation"]["targets"] = ["attn.qkv"]
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "LoRA targets differ"):
                CSGORDTRunner.from_pretrained(temp, dtype=torch.float32)


if __name__ == "__main__":
    unittest.main()
