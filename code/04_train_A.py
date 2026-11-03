# PHC-351 · Hiding a Voice · Bhavishay (24324008)
# 04_train_A.py: Training encoder A: Adam 1e-3, cross-entropy, 8 passes, saved to Drive.
# The code below is the project's Colab notebook cell(s) for this stage, as run (Colab, T4 GPU), in run order.

import time, torch.nn as nn

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
lossf = nn.CrossEntropyLoss()
EPOCHS = 8

for ep in range(EPOCHS):
    model.train(); t0 = time.time()
    tot = correct = 0; running = 0.0
    for xb, yb in train_dl:
        xb, yb = xb.to(dev), yb.to(dev)
        _, logits = model(xb)
        loss = lossf(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()

        running += loss.item() * yb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        tot     += yb.size(0)

    print(f"epoch {ep+1}/{EPOCHS} | loss {running/tot:.3f} "
          f"| train acc {100*correct/tot:.1f}% | {time.time()-t0:.0f}s")
    torch.save(model.state_dict(), '/content/drive/MyDrive/modelA.pt')
