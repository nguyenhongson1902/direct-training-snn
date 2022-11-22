def convert(imgf, labelf, outf, n):
    """
        Converts MNIST dataset to csv files.
        Args:
            imgf (str): Input images file.
            labelf (str): Labels file.
            outf (str): Output csv file.
            n (int): The number of images.
    """
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

convert("./datasets/MNIST/raw/train-images-idx3-ubyte", "./datasets/MNIST/raw/train-labels-idx1-ubyte",
        "./datasets/MNIST/raw/mnist_train.csv", 60000)
convert("./datasets/MNIST/raw/t10k-images-idx3-ubyte", "./datasets/MNIST/raw/t10k-labels-idx1-ubyte",
        "./datasets/MNIST/raw/mnist_test.csv", 10000)