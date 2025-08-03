import json
import matplotlib.pyplot as plt


INPUT_FILE = 'C:/home/myproj/genesis/UR5/ur5/data/ur5_log.json'
OUTPUT_FILE = 'C:/home/myproj/genesis/UR5/ur5/data/ur5_log.png'

def main():
    xs = []
    ys = []
    zs = []
    gs = []

    with open(INPUT_FILE, 'r') as f:
        j = json.loads(f.read())
        for d in j["data"]:
            x, y, z = d["eepos"]
            g = d["is_gripper_closed"]
            xs.append(x)
            ys.append(y)
            zs.append(z)
            gs.append(g)

    # 折れ線グラフでzとgを出力
    plt.plot(xs)
    plt.plot(ys)
    plt.plot(zs)
    plt.plot(gs)
    plt.xlabel('time')
    plt.ylabel('x, y, z, g')
    plt.legend(['x', 'y', 'z', 'g'])
    plt.savefig(OUTPUT_FILE)
    print("write to ", OUTPUT_FILE)



if __name__ == '__main__':
    main()