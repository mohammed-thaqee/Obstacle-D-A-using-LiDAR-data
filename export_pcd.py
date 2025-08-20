import os, json
import pandas as pd

def write_pcd_ascii(points, path):
    """
    Write ASCII PCD file with x y z points.
    """
    n = len(points)
    header = [
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        "FIELDS x y z",
        "SIZE 4 4 4",
        "TYPE F F F",
        "COUNT 1 1 1",
        f"WIDTH {n}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {n}",
        "DATA ascii"
    ]
    with open(path, "w") as f:
        f.write("\n".join(header) + "\n")
        for (x, y, z) in points:
            f.write(f"{x} {y} {z}\n")

def export_all_pcd(csv_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    for idx, row in df.iterrows():
        pts = json.loads(row["lidar_scan_data"])
        write_pcd_ascii(pts, os.path.join(out_dir, f"scan_{idx:04d}.pcd"))
    print(f"Exported {len(df)} PCD files to {out_dir}")

if __name__ == "__main__":
    export_all_pcd("data3d/train.csv", "data3d/pcd/train")
    export_all_pcd("data3d/test.csv", "data3d/pcd/test")
