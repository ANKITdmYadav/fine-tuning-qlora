import json
import matplotlib.pyplot as plt
TRAINER_STATE = "C:\\Users\\komma\\Desktop\\QLORA\\Medical_Qwen\\trainer_state.json"
OUTPUT_IMAGE = "C:\\Users\\komma\\Desktop\\QLORA\\output\\training_loss.png"
with open(TRAINER_STATE, "r") as f:
    state = json.load(f)
history = state["log_history"]
steps = []
losses = []
for item in history:
    if "loss" in item:
        steps.append(item["step"])
        losses.append(item["loss"])
plt.figure(figsize=(8,5))
plt.plot(steps,losses,marker="o")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.savefig(OUTPUT_IMAGE,dpi=300)
plt.show()
print("Training Curve Saved Successfully")