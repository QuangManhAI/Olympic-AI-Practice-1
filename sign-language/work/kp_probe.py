import sys, time, glob, cv2, numpy as np, torch
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights

dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
min_size = int(sys.argv[1]) if len(sys.argv) > 1 else 448
m = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1,
                              min_size=min_size, max_size=min_size).eval().to(dev)

def read(p):
    c = cv2.VideoCapture(p); f = []
    while True:
        r, x = c.read()
        if not r: break
        f.append(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
    return np.stack(f)

vids = sorted(glob.glob('dataset/train/*/*.mp4'))[::400][:8]
rows = []
t0 = time.time(); nfr = 0
for v in vids:
    fr = read(v); idx = np.linspace(0, len(fr) - 1, 16).astype(int); fr = fr[idx]
    x = torch.from_numpy(fr).permute(0, 3, 1, 2).float().div(255).to(dev)
    with torch.no_grad():
        out = m(list(x))
    nfr += len(fr)
    row = []
    for img, o in zip(fr[::3], out[::3]):
        img = img.copy()
        if len(o['scores']):
            kp = o['keypoints'][0].cpu().numpy()
            for j in [5, 6, 7, 8, 9, 10]:  # shoulders, elbows, wrists
                cv2.circle(img, (int(kp[j, 0]), int(kp[j, 1])), 4, (255, 0, 0) if j in (9, 10) else (0, 255, 0), -1)
        row.append(img)
    rows.append(np.hstack(row))
if dev == 'mps': torch.mps.synchronize()
print(f'min_size={min_size}: {nfr / (time.time() - t0):.1f} frames/s')
cv2.imwrite(sys.argv[2], cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
