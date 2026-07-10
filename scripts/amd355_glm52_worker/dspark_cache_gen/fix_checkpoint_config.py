import json, sys, os
ckpt_dir = sys.argv[1] if len(sys.argv) > 1 else "/data/checkpoints/deepspec/dspark_glm5_2_v9_256exp_clean/step_latest"
config_path = os.path.join(ckpt_dir, "config.json")
c = json.load(open(config_path))
c["architectures"] = ["Glm5ForCausalLMDSpark"]
c.pop("quantization_config", None)
json.dump(c, open(config_path, "w"), indent=2, ensure_ascii=False)
print("Fixed: " + config_path)
print("  architectures = " + str(c["architectures"]))
print("  quantization_config removed = " + str("quantization_config" not in c))
