
import numpy as np
import argparse
import os

def read_camera(sequence,ind):
    file = "%s/cams/%08d_cam.txt"%(sequence,ind)
    f = open(file)

    f.readline().strip()

    row1 = f.readline().strip().split()
    row2 = f.readline().strip().split()
    row3 = f.readline().strip().split()

    M = np.stack(
        (np.array(row1).astype(np.float32), np.array(row2).astype(np.float32), np.array(row3).astype(np.float32)))
    f.readline()
    f.readline()
    f.readline()
    row1 = f.readline().strip().split()
    row2 = f.readline().strip().split()
    row3 = f.readline().strip().split()
    K = np.stack(
        (np.array(row1).astype(np.float32), np.array(row2).astype(np.float32), np.array(row3).astype(np.float32)))

    return (K,M)

def parse_scan(scan_ind,output_cameras_file,blendedMVS_path):
    files = os.listdir('%s/scan%d/cams' % (blendedMVS_path,scan_ind))
    num_cams = len(files) - 1

    cameras_new = {}
    for i in range(num_cams):
        Ki, Mi = read_camera("%s/scan%d" % (blendedMVS_path,scan_ind), int(files[i][:8]))
        curp = np.eye(4).astype(np.float32)
        curp[:3, :] =  Ki @ Mi
        cameras_new['world_mat_%d' % i] = curp.copy()

    np.savez(
        output_cameras_file,
        **cameras_new)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parsing blendedMVS')
    parser.add_argument('--blendedMVS_path', type=str, default="BlendedMVS",
                        help='the blendedMVS path')
    parser.add_argument('--output_cameras_file', type=str, default="cameras.npz",
                        help='the output cameras file')
    parser.add_argument('--scan_ind',type=int,
                        help='Scan id')

    args = parser.parse_args()
    parse_scan(args.scan_ind,args.output_cameras_file,args.blendedMVS_path)
