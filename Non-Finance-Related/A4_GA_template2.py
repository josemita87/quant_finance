import os
import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from PIL import Image
from scipy.spatial import KDTree

################### SESSION 1 ###################

# Function 1: Download MNIST images and save them locally
def download_mnist_images():
    # Removed deprecated `parser` argument
    mnist = fetch_openml("mnist_784", version=1, cache=True, as_frame=False)
    images, labels = mnist.data, mnist.target.astype(int)

    selected_images = images[:500]
    selected_labels = labels[:500]

    folder_name = "mnist_images"
    os.makedirs(folder_name, exist_ok=True)

    for i in range(500):
        img_arr = selected_images[i].reshape(28, 28).astype(np.uint8)
        Image.fromarray(img_arr).save( f"{folder_name}/img_{i}_label_{selected_labels[i]}.png")

    return selected_images, selected_labels


# Function 2: Analyze the first 5 images
def analyze_images(image_folder):
    files = sorted(os.listdir(image_folder))[:5]
    for idx, filename in enumerate(files, start=1):
        with Image.open(os.path.join(image_folder, filename)) as img:
            arr = np.array(img)
            print(f"Image {idx}:")
            print(f"  Filename: {filename}")
            print(f"  Dimensions: {img.size}")
            print(f"  Total pixels: {arr.size}")
            print(f"  Pixel range: {arr.min()}–{arr.max()}")
            # Check what type of channel the image has
            if img.mode == "L":
                print("  Image mode: Grayscale")
            elif img.mode == "RGB":
                print("  Image mode: RGB")
            elif img.mode == "RGBA":
                print("  Image mode: RGBA")
            elif img.mode == "1":
                print("  Image mode: Binary (1-bit pixels, black and white)")
            print("-" * 40)

# Function 3: Crop → resize (preserve aspect ratio) → center in 28×28
def crop_image(image):
    arr = np.array(image)
    rows = np.any(arr > 0, axis=1)
    cols = np.any(arr > 0, axis=0)
    
    # Add 1-pixel padding to avoid cutting off digits
    pad = 1
    r0, r1 = max(0, np.where(rows)[0][[0, -1]][0] - pad), \
             min(arr.shape[0], np.where(rows)[0][[0, -1]][1] + pad)
    c0, c1 = max(0, np.where(cols)[0][[0, -1]][0] - pad), \
             min(arr.shape[1], np.where(cols)[0][[0, -1]][1] + pad)
    
    cropped = arr[r0:r1+1, c0:c1+1]
    canvas = np.zeros((28, 28), dtype=np.uint8)
    h, w = cropped.shape
    y_offset = (28 - h) // 2
    x_offset = (28 - w) // 2
    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = cropped
    return Image.fromarray(canvas)

# Function 4: Plot histograms, optionally separate
def plot_histogram_comparison(images, labels, max_labels_per_figure=5, separate_plots=False):
    unique = np.unique(labels)
    sel_imgs, sel_lbls = [], []
    for lbl in unique:
        idx = np.where(labels == lbl)[0][0]
        sel_imgs.append(images[idx].reshape(28, 28))
        sel_lbls.append(lbl)

    if separate_plots:
        for img, lbl in zip(sel_imgs, sel_lbls):
            hist = np.histogram(img, bins=64, range=(0, 255))[0].astype(np.float32)
            hist[hist == 0] = 1e-5
            plt.figure(figsize=(8, 4))
            plt.plot(np.log(hist), label=f"Label {lbl}")
            plt.title(f"Log-Normalized Histogram for Label {lbl}")
            plt.xlabel("Pixel Intensity")
            plt.ylabel("Log Frequency")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.show()
    else:
        total = len(unique)
        for start in range(0, total, max_labels_per_figure):
            plt.figure(figsize=(12, 6))
            for i in range(start, min(start + max_labels_per_figure, total)):
                img, lbl = sel_imgs[i], sel_lbls[i]
                hist = np.histogram(img, bins=64, range=(0, 255))[0].astype(np.float32)
                hist[hist == 0] = 1e-5
                plt.plot(np.log(hist), label=f"Label {lbl}")
            plt.title("Log-Normalized Pixel Intensity Histograms")
            plt.xlabel("Pixel Intensity Bin")
            plt.ylabel("Log Count")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.show()




################### SESSION 2 ###################


def get_pi_digits():
    pi = np.pi
    pi_str = str(pi).replace('.', '')[:10]
    return [int(digit) for digit in pi_str]

pi_digits = get_pi_digits()


def classify_images_with_kdtree(images, labels):
    vecs = images.reshape(len(images), -1)
    kdt_vec = np.zeros(len(images))
    
    for lbl in np.unique(labels):
        idx = np.where(labels == lbl)[0]
        if len(idx) > 5:  # Need enough samples for meaningful comparison
            tree = KDTree(vecs[idx])
            dists, _ = tree.query(vecs[idx], k=6)  # k=6 to ignore self
            avg_d = dists[:, 1:].mean(axis=1)
            thresh = np.median(avg_d)
            kdt_vec[idx] = (avg_d <= thresh).astype(int)
    
    return kdt_vec

def create_random_individual(images, length, labels, kdt_vec):
    n = len(images)
    idxs = np.random.randint(0, n, size=length)
    return images[idxs], labels[idxs], kdt_vec[idxs]

def selection(pop, fits):
    top2 = np.argsort(fits)[-2:][::-1]
    return pop[top2[0]], pop[top2[1]]

def crossover(p1, p2, L):
    imgs1, lbls1, k1 = p1
    imgs2, lbls2, k2 = p2
    pt = random.randint(1, L - 1)
    return (
        np.concatenate([imgs1[:pt], imgs2[pt:]]),
        np.concatenate([lbls1[:pt], lbls2[pt:]]),
        np.concatenate([k1[:pt], k2[pt:]]),
    )

def mutation(ind, L, images, labels, kdt_vec):
    imgs, lbls, kdt = ind
    i = random.randint(0, L - 1)
    j = random.randint(0, len(images) - 1)
    imgs[i], lbls[i], kdt[i] = images[j], labels[j], kdt_vec[j]
    return imgs, lbls, kdt

def fitness(ind, pi_digits):
    imgs, lbls, kdt = ind
    lbls = np.array(lbls, dtype=int)
    pi_q = sum(lbls[i] == pi_digits[i] for i in range(len(pi_digits)))
    k_q = int(np.sum(kdt))
    return pi_q * 5 + k_q, pi_q, k_q  # Increased weight to 5:1

def plot_best_individual(best, pi_digits):
    imgs, lbls, _ = best
    fig, axs = plt.subplots(2, 5, figsize=(12, 6))
    for i, ax in enumerate(axs.ravel()):
        img = imgs[i]
        disp = img.reshape(28, 28) if img.size == 784 else img
        ax.imshow(disp, cmap="gray")
        ax.axis("off")
        ax.set_title(f"Tgt:{pi_digits[i]}  Lbl:{lbls[i]}")
    plt.suptitle("Best Individual")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_evolution(best_pi, avg_pi, best_kd, avg_kd):
    plt.figure(figsize=(8, 4))
    plt.plot(best_pi, label="Best Pi Quality")  
    plt.plot(avg_pi, label="Avg Pi Quality")
    plt.title("Pi Quality Evolution")
    plt.xlabel("Gen")
    plt.ylabel("Matches")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(best_kd, label="Best KD Score")
    plt.plot(avg_kd, label="Avg KD Score")
    plt.title("KDTree Quality Evolution")
    plt.xlabel("Gen")
    plt.ylabel("Sum Scores")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def genetic_algorithm(images, labels, kdt_vec):
    G, N, L, μ = 100, 50, len(pi_digits), 0.1
    pop = [create_random_individual(images, L, labels, kdt_vec) for _ in range(N)]
    best_pi, avg_pi, best_kd, avg_kd = [], [], [], []

    for _ in range(G):
        fits, p_vals, k_vals = [], [], []
        for ind in pop:
            f, pq, kq = fitness(ind, pi_digits)
            fits.append(f); p_vals.append(pq); k_vals.append(kq)
        best_pi.append(max(p_vals)); avg_pi.append(sum(p_vals)/N)
        best_kd.append(max(k_vals)); avg_kd.append(sum(k_vals)/N)

        p1, p2 = selection(pop, fits)
        new_pop = [p1, p2]
        while len(new_pop) < N:
            child = crossover(p1, p2, L)
            if random.random() < μ:
                child = mutation(child, L, images, labels, kdt_vec)
            new_pop.append(child)
        pop = new_pop

    final = [fitness(ind, pi_digits)[0] for ind in pop]
    best = pop[int(np.argmax(final))]
    plot_evolution(best_pi, avg_pi, best_kd, avg_kd)
    plot_best_individual(best, pi_digits)
    return best




def main():
    # Session 1
    images, labels = download_mnist_images()
    print("Analyzing first 5…")
    analyze_images("mnist_images")
    print("Cropping first 20…")
    cropped = [
        crop_image(Image.fromarray(img.reshape(28, 28).astype(np.uint8)))
        for img in images[:20]
    ]

    print("\nOriginal vs. Cropped (first 4):")
    fig, axes = plt.subplots(2, 4, figsize=(8, 4))
    for i in range(4):
        # original
        with Image.open(f"mnist_images/img_{i}_label_{labels[i]}.png") as orig:
            axes[0, i].imshow(np.array(orig), cmap='gray')
            axes[0, i].set_title("Orig")
            axes[0, i].axis("off")
        # cropped
        axes[1, i].imshow(cropped[i], cmap='gray')
        axes[1, i].set_title("Crop")
        axes[1, i].axis("off")
    plt.tight_layout()
    plt.show()

    print("Plotting histograms…")
    plot_histogram_comparison(
        np.array([np.array(im).flatten() for im in cropped]),
        labels[:20],
        separate_plots=True
    )

    # Session 2
    kdt_vec = classify_images_with_kdtree(images, labels)
    _ = genetic_algorithm(images, labels, kdt_vec)


if __name__ == "__main__":
    main()
