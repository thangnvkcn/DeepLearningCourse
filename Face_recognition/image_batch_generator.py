import numpy as np

def image_batch_generator(images, labels, batch_size):
    labels = np.array(labels)
    while True:
        batch_path = np.random.choice(a=len(labels), size=batch_size // 3)
        input1 = []
        for i in batch_path:
            pos = np.where(labels == labels[i])[0]
            neg = np.where(labels != labels[i])[0]
            k = i
            while k == i:
                k = np.random.choice(pos)
            j = i
            while j == i:
                j = np.random.choice(neg)
            input1.append(images[i])
            input1.append(images[j])
            input1.append(images[k])

        input1 = np.array(input1)
        input = [input1, input1, input1]

        yield (input, np.zeros(1))
