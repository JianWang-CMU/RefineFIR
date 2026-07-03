import argparse
import os
import random

import cv2
import numpy as np
import torch
from torchvision import transforms

from id_loss import IDLoss
from model_ae import Denoiser
from refinefir.degradation import degrade_bgr
from refinefir.mediapipe_warp import mediapipe_warp_face


def load_bgr(path, size):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Could not read image: {}".format(path))
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def np2tensor(image_bgr, transform, device):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return transform(image_rgb).to(device).unsqueeze(0)


def tensor_to_bgr(image):
    if image.is_cuda:
        image = image.cpu()
    if image.dim() == 4:
        image = image[0]
    array = ((image.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).detach().numpy()
    array = (array * 255).round().astype(np.uint8)
    return array[:, :, ::-1]


def put_label(image, label):
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 25), (0, 0, 0), -1)
    cv2.putText(out, label[:38], (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="RefineFIR inference with a MediaPipe face warp fallback.")
    parser.add_argument("--input", required=True, help="Aligned target/input face image.")
    parser.add_argument("--reference", required=True, help="Aligned reference face image.")
    parser.add_argument("--checkpoint", required=True, help="RefineFIR generator checkpoint, e.g. latest.pt.")
    parser.add_argument("--id-weight", required=True, help="ArcFace/IR-SE50 identity encoder checkpoint.")
    parser.add_argument("--output", required=True, help="Output restored image path.")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-degrade", action="store_true", help="Use input directly as LQ instead of synthesizing degradation.")
    parser.add_argument("--noise-min", type=float, default=0.0)
    parser.add_argument("--noise-max", type=float, default=50.0)
    parser.add_argument("--save-debug", action="store_true", help="Also save LQ, warp, landmarks, and comparison strip.")
    parser.add_argument("--triangles", default="assets/canonical_face_mesh/triangles.txt")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    source = load_bgr(args.reference, args.size)
    target = load_bgr(args.input, args.size)
    if args.skip_degrade:
        lq = target
    else:
        lq = degrade_bgr(target, size=args.size, seed=args.seed, noise_range=(args.noise_min, args.noise_max))

    warp, landmarks = mediapipe_warp_face(source, lq, args.triangles)

    generator = Denoiser(args.size, 512, 8, channel_multiplier=2).to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    generator.load_state_dict(ckpt["g"] if "g" in ckpt else ckpt, strict=True)

    id_encoder = IDLoss(args.id_weight).to(device).eval()

    lq_t = np2tensor(lq, transform, device)
    source_t = np2tensor(source, transform, device)
    warp_t = np2tensor(warp, transform, device)

    with torch.no_grad():
        target_id = id_encoder.encode(source_t.mean(1, keepdim=True).repeat(1, 3, 1, 1))
        result_t = generator(torch.cat([lq_t, warp_t], 1), target_id)

    result = tensor_to_bgr(result_t)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    cv2.imwrite(args.output, result)

    if args.save_debug:
        stem, ext = os.path.splitext(args.output)
        cv2.imwrite(stem + "_lq" + ext, lq)
        cv2.imwrite(stem + "_warp_mediapipe" + ext, warp)
        cv2.imwrite(stem + "_landmarks_mediapipe" + ext, landmarks)
        strip = cv2.hconcat(
            [
                put_label(lq, "input / LQ"),
                put_label(source, "reference"),
                put_label(landmarks, "MediaPipe landmarks"),
                put_label(warp, "MediaPipe warp"),
                put_label(result, "RefineFIR"),
            ]
        )
        cv2.imwrite(stem + "_comparison" + ext, strip)

    print(args.output)


if __name__ == "__main__":
    main()

