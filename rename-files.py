import os

folder = "trainingimages"
files = os.listdir(folder)
max_n = 0
for f in files:
    file_name, ending = f.split(".")
    img, nr = file_name.split("_")
    try:
        nr = int(nr)
    except:
        print(f"deleted this loser {f}")
        os.remove(f"{folder}/{f}")
        break
    new_nr = int(nr) + 2400
    os.rename(f"{folder}/{f}", f"{folder}/img_{new_nr}.jpeg")


print("done renaming")