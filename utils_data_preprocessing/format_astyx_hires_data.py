import numpy as np

# UNOFORMATTED_DATA_FILE = "/home/khatch/Documents/Bayesian-Dynamics/datasets/radar_dataset_astyx_hires2019/radar_6455/000000.txt"
# FORMATTED_DATA_PATH = "/home/khatch/Documents/Bayesian-Dynamics/datasets/radar_dataset_astyx_hires2019/formatted/000000.txt"
UNOFORMATTED_DATA_FILE = "/Users/khatch/Desktop/SISL/Bayesian Hilbert Project/radar_dataset_astyx_hires2019/dataset_astyx_hires2019/dataset_astyx_hires2019/radar_6455/000047.txt"
FORMATTED_DATA_PATH = "/Users/khatch/Desktop/SISL/Bayesian Hilbert Project/Bayesian-Dynamics/datasets/radar_dataset_astyx_hires2019/formatted/000047.csv"
NORMALIZED_DATA_PATH = "/Users/khatch/Desktop/SISL/Bayesian Hilbert Project/Bayesian-Dynamics/datasets/radar_dataset_astyx_hires2019/normalized/000047.csv"


"""
Normalize the data
"""

def run_format(unformatted_data_file, formatted_data_path, normalized_data_path):
    # unformatted_data = np.loadtxt(unformatted_data_file, delimiter=" ", skiprows=2)
    # print("unformatted_data:", unformatted_data)
    # print("unformatted_data.shape:", unformatted_data.shape)

    new_file_contents = ""

    i = -1
    with open(unformatted_data_file, "r") as f:
        next(f)
        for line in f:
            line = line.replace("\n", "")
            line = line.split(" ")
            if i < 0:
                line = line[:-2]
                line += ["V_X", "V_Y", "V_Z"]
                line.insert(0, "t")

                # new_file_contents += "t," + line.replace(" ", ",")
            else:
                line = line[:-1]
                vr = line[-1]
                line += [vr, vr]
                line.insert(0, "0")
                # new_file_contents += "{},".format(i) + line.replace(" ", "")
                # new_file_contents += "{},".format(0) + line.replace(" ", ",")

            new_line = ",".join(line)
            new_line += "\n"
            new_file_contents += new_line
            i += 1

    with open(formatted_data_path,  "w") as f:
        f.write(new_file_contents)



    X = np.loadtxt(formatted_data_path, delimiter=",", skiprows=1)#, usecols=list(range(1, 7)))
    print("X.shape:", X.shape)
    for col in range(1, X.shape[1]):
        min_val = np.amin(X[:, col])
        max_val = np.amax(X[:, col])
        print("\nmin_val:", min_val)
        print("max_val:", max_val)
        X[:, col] -= min_val
        X[:, col] /= (max_val - min_val)
        X[:, col] *= 2
        X[:, col] -= 1

    for col in range(1, X.shape[1]):
        min_val = np.amin(X[:, col])
        max_val = np.amax(X[:, col])
        print("\nmin_val:", min_val)
        print("max_val:", max_val)

    header = get_header(formatted_data_path)
    np.savetxt(normalized_data_path, X, delimiter=",", header=header, comments="")


# def normalize(val, max, min):
#     val -= min
#     val /= (max - min)
#     val *= 2
#     val -= 1
#     return val
#
# def unnormalize(val, max, min):
#     val += 1
#     val /= 2
#     val *= (max - min)
#     val += min
#     return val

def get_header(data_path):
    with open(data_path, "r") as f:
        line = f.readline()
        header = line.replace("\n", "")#.split(",")
        return header

if __name__ == "__main__":
    run_format(UNOFORMATTED_DATA_FILE, FORMATTED_DATA_PATH, NORMALIZED_DATA_PATH)
