import socket, struct, pickle, numpy as np, threading


# with np.load('merged_dataset.npz') as data:
#     print(data.keys())
#     print(data["images"].shape)
#     print(data["labels"].shape)

#     from sklearn.model_selection import train_test_split
#     x_train, x_test, y_train, y_test = train_test_split(data["images"], data["labels"], test_size=0.2, random_state=42, stratify=data["labels"])
#     print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

#     np.savez('custom_mnist.npz', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
#     print("Saved custom_mnist.npz file")

# with np.load('datasets/custom_mnist.npz') as data:
#     x_train, y_train = data['x_train'], data['y_train']
#     x_test, y_test = data['x_test'], data['y_test']

#     y_train_ascii = np.array([ord(c) for c in y_train])
#     y_test_ascii = np.array([ord(c) for c in y_test])

#     train_mask = y_train_ascii < 58  # ASCII for '9' is 57
#     test_mask = y_test_ascii < 58

#     x_train = x_train[train_mask]
#     y_train = y_train_ascii[train_mask] - 48  # Convert ASCII to int (e.g., '0'->0)
    
#     x_test = x_test[test_mask]
#     y_test = y_test_ascii[test_mask] - 48

#     np.savez('datasets/custom_mnist.npz', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
#     print("Saved custom_mnist.npz file")

   


# with np.load('datasets/mnist.npz') as data:
#     x_train, y_train = data['x_train'], data['y_train']
#     x_test, y_test = data['x_test'], data['y_test']

#     print(np.unique(y_train))

#     print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)



with np.load('datasets/custom_mnist.npz') as data:
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']

    x_train = x_train[:60000]
    y_train = y_train[:60000]
    x_test = x_test[:10000]
    y_test = y_test[:10000]

    np.savez('datasets/custom_mnist.npz', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    print("Saved custom_mnist.npz file")
