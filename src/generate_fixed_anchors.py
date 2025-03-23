import numpy as np
import torch
import argparse

def init_points(dim, num_points, radius):
    points = np.random.randn(num_points, dim)
    points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
    points *= np.random.uniform(0, radius, size=(num_points, 1))

    return points


def compute_pairwise_dist(points):
    num_points = len(points)
    distances = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(i + 1, num_points):
            distances[i][j] = distances[j][i] = np.linalg.norm(points[i] - points[j])
    return distances


def repulsion_forces(points, distances):
    num_points = len(points)
    forces = np.zeros_like(points)
    for i in range(num_points):
        for j in range(i + 1, num_points):
            if distances[i][j] > 0:
                direction = (points[i] - points[j]) / distances[i][j]
                force_magnitude = 1 / (distances[i][j] ** 2)
                forces[i] += force_magnitude * direction
                forces[j] -= force_magnitude * direction
    return forces


def project_to_sphere(points, radius):
    norms = np.linalg.norm(points, axis=1)
    points = points * np.minimum(1, radius / norms)[:, np.newaxis]
    return points


def optimize_points(dim, num_points, radius, steps=1000, lr=0.01):
    points = init_points(dim, num_points, radius)

    for step in range(steps):
        dists = compute_pairwise_dist(points)
        forces = repulsion_forces(points, dists)

        points += lr*forces

        points = project_to_sphere(points, radius)
        
        # if step % 100 == 0:
        #     min_dist = np.min(dists[np.nonzero(dists)])
        #     print(f"Step {step}: Min dist = {min_dist:.5f}")
    
    return points


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=1024)
    parser.add_argument('--radius', type=float, default=50.)
    parser.add_argument('--save_path', type=str, default='./emo_anchors/fixed_anchors')
    args = parser.parse_args()

    meld_anchors = optimize_points(args.dim, 7, args.radius, steps=10000)
    iemocap_anchors = optimize_points(args.dim, 6, args.radius, steps=10000)
    iemocap4_anchors = optimize_points(args.dim, 4, args.radius, steps=10000)

    meld_anchors = torch.tensor(meld_anchors, dtype=torch.float32)
    iemocap_anchors = torch.tensor(iemocap_anchors, dtype=torch.float32)
    iemocap4_anchors = torch.tensor(iemocap4_anchors, dtype=torch.float32)

    torch.save(meld_anchors, f'{args.save_path}/meld_emo.pt')
    torch.save(iemocap_anchors, f'{args.save_path}/iemocap_emo.pt')
    torch.save(iemocap4_anchors, f'{args.save_path}/iemocap4_emo.pt')