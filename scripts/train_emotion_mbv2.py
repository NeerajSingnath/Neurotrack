# D:\Neurotrack\scripts\train_emotion_mbv2.py
import torch, json
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path

def main():
    ROOT = Path(__file__).resolve().parent.parent
    FER_DIR = ROOT / "datasets" / "fer2013"
    MODELS_DIR = ROOT / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    BATCH_SIZE = 64
    EPOCHS = 20
    LR = 1e-3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Transforms (ImageNet norm)
    train_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.2,0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    # FER-2013 in ImageFolder layout:
    # fer2013/train/<class>/*.jpg, fer2013/val/<class>/*.jpg
    train_ds = datasets.ImageFolder(FER_DIR / "train", transform=train_tf)
    val_ds   = datasets.ImageFolder(FER_DIR / "val",   transform=val_tf)

    # Windows-safe: workers=0 to avoid spawn issues
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=(DEVICE=="cuda"))
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=(DEVICE=="cuda"))

    classes = train_ds.classes
    with open(MODELS_DIR / "emotion_labels.json", "w") as f:
        json.dump(classes, f)
    print(f"[OK] Saved label map -> {MODELS_DIR / 'emotion_labels.json'}")

    # Pretrained MobileNetV2 + new head
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    in_feats = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_feats, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, len(classes))
    )
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    best_path = MODELS_DIR / "emotion_mbv2_best.pt"

    for epoch in range(1, EPOCHS+1):
        # ---- train ----
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(DEVICE=="cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()*y.size(0)
            pred = logits.argmax(1)
            correct += (pred==y).sum().item()
            total += y.size(0)
        train_loss = loss_sum/total
        train_acc = correct/total

        # ---- validate ----
        model.eval()
        total, correct, val_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss_sum += loss.item()*y.size(0)
                pred = logits.argmax(1)
                correct += (pred==y).sum().item()
                total += y.size(0)
        val_loss = val_loss_sum/total
        val_acc  = correct/total

        scheduler.step()

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"train_loss={len(str(round(train_loss,4))).__str__() and train_loss:.4f} acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"[OK] Saved best -> {best_path} (val_acc={best_acc:.3f})")

    print("[DONE] Best val_acc:", best_acc)

if __name__ == "__main__":
    main()
