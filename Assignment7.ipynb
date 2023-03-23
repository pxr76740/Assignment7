{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU",
    "gpuClass": "standard"
  },
  "cells": [
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "from keras.datasets import cifar10\n",
        "from keras.models import Sequential\n",
        "from keras.layers import Dense, Dropout, Flatten\n",
        "from keras.constraints import maxnorm\n",
        "from keras.optimizers import SGD\n",
        "from keras.layers.convolutional import Conv2D, MaxPooling2D\n",
        "from keras.utils import np_utils"
      ],
      "metadata": {
        "id": "MtZG9dJzw4ld"
      },
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "np.random.seed(7)"
      ],
      "metadata": {
        "id": "n4lci1f3w8hC"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "(X_train, y_train), (X_test, y_test) = cifar10.load_data()"
      ],
      "metadata": {
        "id": "m4bzJoVOxBX_"
      },
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X_train = X_train.astype('float32') / 255.0\n",
        "X_test = X_test.astype('float32') / 255.0"
      ],
      "metadata": {
        "id": "3DvFG935xFI2"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "y_train = np_utils.to_categorical(y_train)\n",
        "y_test = np_utils.to_categorical(y_test)\n",
        "num_classes = y_test.shape[1]"
      ],
      "metadata": {
        "id": "i_R7uAN4xIrm"
      },
      "execution_count": 9,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model = Sequential()\n",
        "model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))\n",
        "model.add(Dropout(0.2))\n",
        "model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))\n",
        "model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))\n",
        "model.add(Flatten())\n",
        "model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))\n",
        "model.add(Dropout(0.5))\n",
        "model.add(Dense(num_classes, activation='softmax'))\n"
      ],
      "metadata": {
        "id": "z_NGevkcxMSf"
      },
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "sgd = SGD(learning_rate=0.01, momentum=0.9, decay=1e-6)\n",
        "model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])\n",
        "print(model.summary())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "rhzjj8FaxN77",
        "outputId": "0b033a14-600d-4cf1-8515-ce5515d02091"
      },
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model: \"sequential_1\"\n",
            "_________________________________________________________________\n",
            " Layer (type)                Output Shape              Param #   \n",
            "=================================================================\n",
            " conv2d_2 (Conv2D)           (None, 32, 32, 32)        896       \n",
            "                                                                 \n",
            " dropout_2 (Dropout)         (None, 32, 32, 32)        0         \n",
            "                                                                 \n",
            " conv2d_3 (Conv2D)           (None, 32, 32, 32)        9248      \n",
            "                                                                 \n",
            " max_pooling2d_1 (MaxPooling  (None, 16, 16, 32)       0         \n",
            " 2D)                                                             \n",
            "                                                                 \n",
            " flatten_1 (Flatten)         (None, 8192)              0         \n",
            "                                                                 \n",
            " dense_2 (Dense)             (None, 512)               4194816   \n",
            "                                                                 \n",
            " dropout_3 (Dropout)         (None, 512)               0         \n",
            "                                                                 \n",
            " dense_3 (Dense)             (None, 10)                5130      \n",
            "                                                                 \n",
            "=================================================================\n",
            "Total params: 4,210,090\n",
            "Trainable params: 4,210,090\n",
            "Non-trainable params: 0\n",
            "_________________________________________________________________\n",
            "None\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "epochs = 5\n",
        "batch_size = 32\n",
        "model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "COpaIeenxXuH",
        "outputId": "89ce2eea-a785-4f62-d7bc-567c0b2c98af"
      },
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/5\n",
            "1563/1563 [==============================] - 19s 7ms/step - loss: 1.7232 - accuracy: 0.3746 - val_loss: 1.4776 - val_accuracy: 0.4563\n",
            "Epoch 2/5\n",
            "1563/1563 [==============================] - 10s 6ms/step - loss: 1.3675 - accuracy: 0.5117 - val_loss: 1.2470 - val_accuracy: 0.5551\n",
            "Epoch 3/5\n",
            "1563/1563 [==============================] - 10s 6ms/step - loss: 1.2071 - accuracy: 0.5716 - val_loss: 1.1232 - val_accuracy: 0.6047\n",
            "Epoch 4/5\n",
            "1563/1563 [==============================] - 10s 7ms/step - loss: 1.0855 - accuracy: 0.6136 - val_loss: 1.1554 - val_accuracy: 0.5928\n",
            "Epoch 5/5\n",
            "1563/1563 [==============================] - 10s 7ms/step - loss: 0.9709 - accuracy: 0.6583 - val_loss: 0.9986 - val_accuracy: 0.6550\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7f689d6d65e0>"
            ]
          },
          "metadata": {},
          "execution_count": 12
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "scores = model.evaluate(X_test, y_test, verbose=0)\n",
        "print(\"Accuracy: %.2f%%\" % (scores[1]*100))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "gymoyEPsxpd3",
        "outputId": "ac10174d-d62f-4f23-9766-43fe8b545e9a"
      },
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy: 65.50%\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "from keras.datasets import cifar10\n",
        "from keras.models import Sequential\n",
        "from keras.layers import Dense, Dropout, Flatten\n",
        "from keras.layers.convolutional import Conv2D, MaxPooling2D\n",
        "from keras.constraints import maxnorm\n",
        "from keras.utils import np_utils\n",
        "from keras.optimizers import SGD\n",
        "\n",
        "# Fix random seed for reproducibility\n",
        "np.random.seed(7)\n",
        "\n",
        "# Load data\n",
        "(X_train, y_train), (X_test, y_test) = cifar10.load_data()\n",
        "\n",
        "# Normalize inputs from 0-255 to 0.0-1.0\n",
        "X_train = X_train.astype('float32') / 255.0\n",
        "X_test = X_test.astype('float32') / 255.0\n",
        "\n",
        "# One hot encode outputs\n",
        "y_train = np_utils.to_categorical(y_train)\n",
        "y_test = np_utils.to_categorical(y_test)\n",
        "num_classes = y_test.shape[1]\n",
        "\n",
        "# Create the model\n",
        "model = Sequential()\n",
        "model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))\n",
        "model.add(Dropout(0.2))\n",
        "model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))\n",
        "model.add(MaxPooling2D(pool_size=(2, 2)))\n",
        "model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))\n",
        "model.add(Dropout(0.2))\n",
        "model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))\n",
        "model.add(MaxPooling2D(pool_size=(2, 2)))\n",
        "model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))\n",
        "model.add(Dropout(0.2))\n",
        "model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))\n",
        "model.add(MaxPooling2D(pool_size=(2, 2)))\n",
        "model.add(Flatten())\n",
        "model.add(Dropout(0.2))\n",
        "model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))\n",
        "model.add(Dropout(0.2))\n",
        "model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))\n",
        "model.add(Dropout(0.2))\n",
        "model.add(Dense(num_classes, activation='softmax'))\n",
        "\n",
        "# Compile model\n",
        "epochs = 5\n",
        "learning_rate = 0.01\n",
        "decay_rate = learning_rate / epochs\n",
        "sgd = SGD(lr=learning_rate, momentum=0.9, decay=decay_rate, nesterov=False)\n",
        "model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])\n",
        "print(model.summary())\n",
        "\n",
        "# Fit the model\n",
        "history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)\n",
        "\n",
        "# Evaluate the model\n",
        "scores = model.evaluate(X_test, y_test, verbose=0)\n",
        "print(\"Accuracy: %.2f%%\" % (scores[1] * 100))\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-Lc36Iq-xsa7",
        "outputId": "68bb1447-a60a-4f88-de1e-4ee99675a3f8"
      },
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model: \"sequential_2\"\n",
            "_________________________________________________________________\n",
            " Layer (type)                Output Shape              Param #   \n",
            "=================================================================\n",
            " conv2d_4 (Conv2D)           (None, 32, 32, 32)        896       \n",
            "                                                                 \n",
            " dropout_4 (Dropout)         (None, 32, 32, 32)        0         \n",
            "                                                                 \n",
            " conv2d_5 (Conv2D)           (None, 32, 32, 32)        9248      \n",
            "                                                                 \n",
            " max_pooling2d_2 (MaxPooling  (None, 16, 16, 32)       0         \n",
            " 2D)                                                             \n",
            "                                                                 \n",
            " conv2d_6 (Conv2D)           (None, 16, 16, 64)        18496     \n",
            "                                                                 \n",
            " dropout_5 (Dropout)         (None, 16, 16, 64)        0         \n",
            "                                                                 \n",
            " conv2d_7 (Conv2D)           (None, 16, 16, 64)        36928     \n",
            "                                                                 \n",
            " max_pooling2d_3 (MaxPooling  (None, 8, 8, 64)         0         \n",
            " 2D)                                                             \n",
            "                                                                 \n",
            " conv2d_8 (Conv2D)           (None, 8, 8, 128)         73856     \n",
            "                                                                 \n",
            " dropout_6 (Dropout)         (None, 8, 8, 128)         0         \n",
            "                                                                 \n",
            " conv2d_9 (Conv2D)           (None, 8, 8, 128)         147584    \n",
            "                                                                 \n",
            " max_pooling2d_4 (MaxPooling  (None, 4, 4, 128)        0         \n",
            " 2D)                                                             \n",
            "                                                                 \n",
            " flatten_2 (Flatten)         (None, 2048)              0         \n",
            "                                                                 \n",
            " dropout_7 (Dropout)         (None, 2048)              0         \n",
            "                                                                 \n",
            " dense_4 (Dense)             (None, 1024)              2098176   \n",
            "                                                                 \n",
            " dropout_8 (Dropout)         (None, 1024)              0         \n",
            "                                                                 \n",
            " dense_5 (Dense)             (None, 512)               524800    \n",
            "                                                                 \n",
            " dropout_9 (Dropout)         (None, 512)               0         \n",
            "                                                                 \n",
            " dense_6 (Dense)             (None, 10)                5130      \n",
            "                                                                 \n",
            "=================================================================\n",
            "Total params: 2,915,114\n",
            "Trainable params: 2,915,114\n",
            "Non-trainable params: 0\n",
            "_________________________________________________________________\n",
            "None\n",
            "Epoch 1/5\n",
            "1563/1563 [==============================] - 15s 9ms/step - loss: 1.9322 - accuracy: 0.2796 - val_loss: 1.6108 - val_accuracy: 0.4168\n",
            "Epoch 2/5\n",
            "1563/1563 [==============================] - 13s 9ms/step - loss: 1.5375 - accuracy: 0.4379 - val_loss: 1.4261 - val_accuracy: 0.4795\n",
            "Epoch 3/5\n",
            "1563/1563 [==============================] - 13s 9ms/step - loss: 1.3979 - accuracy: 0.4918 - val_loss: 1.3406 - val_accuracy: 0.5164\n",
            "Epoch 4/5\n",
            "1563/1563 [==============================] - 13s 8ms/step - loss: 1.3128 - accuracy: 0.5217 - val_loss: 1.2901 - val_accuracy: 0.5367\n",
            "Epoch 5/5\n",
            "1563/1563 [==============================] - 13s 9ms/step - loss: 1.2504 - accuracy: 0.5459 - val_loss: 1.1804 - val_accuracy: 0.5735\n",
            "Accuracy: 57.35%\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Predict the first 4 images of the test data\n",
        "predictions = model.predict(X_test[:4])\n",
        "# Convert the predictions to class labels\n",
        "predicted_labels = numpy.argmax(predictions, axis=1)\n",
        "# Convert the actual labels to class labels\n",
        "actual_labels = numpy.argmax(y_test[:4], axis=1)\n",
        "\n",
        "# Print the predicted and actual labels for the first 4 images\n",
        "print(\"Predicted labels:\", predicted_labels)\n",
        "print(\"Actual labels:   \", actual_labels)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vvcyfHmUzJ2n",
        "outputId": "a39a54f1-43d3-4e60-aa47-abeedd908e1b"
      },
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "1/1 [==============================] - 0s 21ms/step\n",
            "Predicted labels: [3 8 8 8]\n",
            "Actual labels:    [3 8 8 0]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# Plot the training and validation loss\n",
        "plt.plot(history.history['loss'])\n",
        "plt.plot(history.history['val_loss'])\n",
        "plt.title('Model Loss')\n",
        "plt.ylabel('Loss')\n",
        "plt.xlabel('Epoch')\n",
        "plt.legend(['train', 'val'], loc='upper right')\n",
        "plt.show()\n",
        "\n",
        "# Plot the training and validation accuracy\n",
        "plt.plot(history.history['accuracy'])\n",
        "plt.plot(history.history['val_accuracy'])\n",
        "plt.title('Model Accuracy')\n",
        "plt.ylabel('Accuracy')\n",
        "plt.xlabel('Epoch')\n",
        "plt.legend(['train', 'val'], loc='lower right')\n",
        "plt.show()\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 573
        },
        "id": "jLt_UBB5zTNk",
        "outputId": "6f238606-4fa3-4d36-8523-65f2ef1cd9a6"
      },
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYgAAAEWCAYAAAB8LwAVAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAAsTAAALEwEAmpwYAAA0yklEQVR4nO3deXiV1bX48e9KCAlDIJAwZiDMswSIKIiIAxYnsNoWtK3a22qdrjj11vZatdbe2vbW1gGrlp+3tnWiWCVOVVFwQpQwyQwJUxLGJCSMgQzr98d+A4dwgJOQkzcnWZ/nyeM573DOyivnrOy9196vqCrGGGNMTVF+B2CMMaZxsgRhjDEmKEsQxhhjgrIEYYwxJihLEMYYY4KyBGGMMSYoSxCm2RORdBFREWkRwrE3iMhnDRGXMX6zBGEiiohsEpHDIpJUY/sS70s+3afQAmNpKyL7RORdv2Mx5nRYgjCRaCNwTfUTERkKtPYvnONcDRwCJohI14Z841BaQcaEyhKEiUR/B64LeH498LfAA0SkvYj8TUR2ichmEblfRKK8fdEi8r8iUigiG4DLgpz7/0Rkm4gUiMgjIhJdi/iuB54Bvga+V+O1x4rIfBEpEZE8EbnB295KRP7gxVoqIp9528aLSH6N19gkIhd5jx8SkVki8g8R2QPcICKjROQL7z22ichTItIy4PzBIvKBiBSLyA4R+bmIdBWRAyKSGHDcCO/6xdTidzdNiCUIE4kWAO1EZKD3xT0V+EeNY54E2gO9gPNwCeUH3r4bgcuB4UAm8K0a5/4VqAD6eMdcDPwolMBEpAcwHnjR+7muxr53vdg6ARnAUm/3/wIjgTFAR+C/gKpQ3hOYDMwCErz3rATuApKA0cCFwK1eDPHAHODfQHfvd/xQVbcD84DvBLzu94FXVLU8xDhME2MJwkSq6lbEBGA1UFC9IyBp/ExV96rqJuAPuC88cF+Cf1LVPFUtBn4TcG4X4FLgTlXdr6o7gT96rxeK7wNfq+oq4BVgsIgM9/ZdC8xR1ZdVtVxVi1R1qdey+Q9gmqoWqGqlqs5X1UMhvucXqvqGqlap6kFVXaSqC1S1wvvdn8UlSXCJcbuq/kFVy7zr86W37wW8Fo93Da/BXWfTTFl/pYlUfwc+AXpSo3sJ95dzDLA5YNtmINl73B3Iq7GvWg/v3G0iUr0tqsbxJ3Md8BcAVS0QkY9xXU5LgFQgN8g5SUDcCfaF4pjYRKQf8BiuddQa9zlf5O0+UQwAs4FnRKQn0B8oVdWv6hiTaQKsBWEikqpuxg1WXwr8q8buQqAc92VfLY2jrYxtuC/KwH3V8nADzEmqmuD9tFPVwaeKSUTGAH2Bn4nIdhHZDpwFXOsNHucBvYOcWgiUnWDffgIG4L2/7DvVOKbmksx/BtYAfVW1HfBzoDrb5eG63Y6jqmXATFwr4vtY66HZswRhItkPgQtUdX/gRlWtxH3R/VpE4r2+/7s5Ok4xE7hDRFJEpANwX8C524D3gT+ISDsRiRKR3iJyHqd2PfABMAg3vpABDAFaAZfgxgcuEpHviEgLEUkUkQxVrQKeBx4Tke7eIPpoEYkF1gFxInKZN1h8PxB7ijjigT3APhEZANwSsO8toJuI3Ckisd71OStg/9+AG4BJWIJo9ixBmIilqrmqmn2C3f+J++t7A/AZ8BLuSxhcF9B7wDJgMce3QK4DWgKrgN24AeBuJ4tFROJwYxtPqur2gJ+NuC/a61V1C67Fcw9QjBugHua9xL3AcmCht++3QJSqluIGmGfgWkD7gWOqmoK4Fzfesdf7XV+t3qGqe3HjNlcA24H1wPkB+z/HDY4v9lppphkTu2GQMSaQiHwEvKSqM/yOxfjLEoQx5ggRORPXTZbqtTZMM2ZdTMYYAETkBdwciTstORiwFoQxxpgTsBaEMcaYoJrMRLmkpCRNT0/3OwxjjIkoixYtKlTVmnNrgCaUINLT08nOPlHFozHGmGBE5ITlzNbFZIwxJihLEMYYY4KyBGGMMSYoSxDGGGOCsgRhjDEmKEsQxhhjgrIEYYwxJqgmMw/CGGOanQPFsPYdqCyHzB+c+vhasgRhjDGRZN9OWPMWrMqCjZ+AVkLKmZYgjDGmWdqzFVa/6ZLClvmgVdCxF4z5Txg0CbqPCMvbWoIwxpjGaPdmWJ3lkkL+V25bpwFw7r0waDJ0GQwiJ3+N02QJwhhjGovCHFg92yWFbUvdtq5D4YL7YeBk6NSvQcOxBGGMMX5RhZ2rvZbCbNi5ym1PHgkTHoaBV7iuJJ9YgjDGmIakCtuWuYSwOguKcgCBtNEw8VGXFNqn+B0lYAnCGGPCr6oKChbBqjdcUijZAhIN6WPh7FtgwBUQ38XvKI9jCcIYY8KhqhK2LPBaCm/C3q0QFQO9xsO4n0D/y6BNot9RnpQlCGOMqS+V5bDpUzfIvOYt2L8LomOhz0Uw6EHoNxFaJfgdZcgsQRhjzOmoOAQb5rmksPZtOLgbYlpD34vdHIW+F0NsvN9R1oklCGOMqa3yg5AzxyWFdf+GQ3sgtp1rIQyaBL0vhJat/Y7ytFmCMMaYUBzaC+vfd0lh/ftQfgBadYCBk9zEtV7nQYtYv6OsV5YgjDHmRA6WuBbCqizXYqg8BG06wbCpLjGkj4XoGL+jDJuwJggRmQg8DkQDM1T10Rr7bwB+DxR4m55S1Rnevkpgubd9i6pOCmesxhgDwP4iN5awKsuNLVSVQ3x3txjewEmQdjZERfsdZYMIW4IQkWhgOjAByAcWikiWqq6qceirqnp7kJc4qKoZ4YrPGGOO2LsD1rzpSlI3fe5WSE3oAWff7Ja4SB4JUc3v9jnhbEGMAnJUdQOAiLwCTAZqJghjjGl4pfneCqmz3XwFFBL7wtg7XUuh27CwL4bX2IUzQSQDeQHP84Gzghx3tYiMA9YBd6lq9TlxIpINVACPquobNU8UkZuAmwDS0tLqMXRjTJNUvMF1Ha3OcjObAToPhvH3uYHmTgOafVII5Pcg9ZvAy6p6SER+DLwAXODt66GqBSLSC/hIRJaram7gyar6HPAcQGZmpjZk4MaYCLFrrZcUZsN2b1izWwZc+IDrPkrq42t49aGySomOqv/EFs4EUQCkBjxP4ehgNACqWhTwdAbwu4B9Bd5/N4jIPGA4cEyCMMaY46jCjpVHF8PbtcZtTxkFF//aLYbXoYe/MZ6G8soq1m7fy9K8EpbmlbAsr4Su7eP4+w+DddCcnnAmiIVAXxHpiUsMU4FrAw8QkW6qus17OglY7W3vABzwWhZJwDkEJA9jjDmGKmxdfLT7qHgDSBT0OAcyfwgDL4d23f2OstZUlfzdB48kgqV5JazYWkpZeRUAHdu0JCM1gTG9w7OmU9gShKpWiMjtwHu4MtfnVXWliDwMZKtqFnCHiEzCjTMUAzd4pw8EnhWRKiAKNwZhg9vGmKOqqtyd1lZlucHm0i0Q1QJ6joMxd8CAy6FtJ7+jrJXSg+V8nV/C0i1e6yC/hMJ9hwGIbRHFkOT2fPesHgxLTWB4agIpHVohYRwzEdWm0XWfmZmp2dnZfodhjAmnygp3T+bqpLBvO0S3hN4XuMqj/pdA645+RxmSwxVVrNm+50hX0dK8Ejbs2n9kf5/ObRmWkkBGmksG/bvGExNd/6W2IrJIVTOD7fN7kNoYY06ushw2fnx0hdQDRdCiFfS9yA0y9/sGxLXzO8qTUlW2FB84Jhms3LqHwxWuqyipbSwZqQlcPSKFYSkJnJHannZx/s/QtgRhjGl8ystgw9yjK6SWlULLti4ZDJrsls9u2cbvKE+o5MDhYwaRl+WXUrzfdRXFxUQxNLk914/uQUZqBzLSEujePi6sXUV1ZQnCGOOvqkoo3gg7Vrjqox0rYOOncHgvxLWH/pd6i+GdDzFxfkd7nEMVlazetpelW3YfSQqbig4AbkpF385tuWhgZ4alJpCRmkD/LvG0CENXUThYgjDGNJwDxV4SWHk0IexcDRUH3X6JcrOZh1zlls1OHwctWvobcwBVZVPRAZbm7WZZXilL8kpYvXUPhytdV1HneNdV9J0zU8lITWBocnviG0FXUV1ZgjDG1L+Kw1C0PiARrHKP9249ekzrJOgyGDL/w/23y2Do1B9iWvkXdw3F+w+zLK+EJQHdRaUHywFo3TKaocnt+cHYdDK8weRu7RtP7PXBEoQxpu5UYd+OgO4h72fXWrcKKrgqo0793f0SqhNBlyHQtrO/sddQVl7Jyq17jhk72FLsuoqiBPp1ieeSIV3JSHXJoG/n+LDMXm5MLEEYY0JTftDNSq7ZRXQgYEGEdskuAfSd4JJAl8GQ2KfR3TOhqkrZULj/mAloq7ftoaLKlf13ax9HRmoC156VdqSrqE1s8/u6bH6/sTHm5FShZMuxiWDnKijKAXV97cS0hs4DYcBlRxNB50GNdg5C4b5DRyafVU9A21tWAUCbltGckZLAjeN6udZBagJd2jW+wXA/WIIwpjkr2+MGiQO7iHaucvdYrtahp0sAg795tHuoQ3qjvWnOwcOVrNhaenTsYEsJBSVuEDw6SujfJZ4rhnU/Mm7Qu1PbJt9VVFeWIIxpDo4rJfVaBiWbjx4T294lgDOmHE0EnQdAbLx/cZ9CVZWSu2vfMYPIa7bvpdLrKkpOaEVGagI3jEknIy2Bwd3b0bqlfe2Fyq6UMU3NgeJj5xTsWAk71xxfSpo8EkZcd7SLqH1Ko78Xws49ZSwJGDf4Or+UfYdcV1F8bAuGpSZw83m9yEjtwLDU9nSOt66i02EJwphIdVwpqdcy2Lvt6DGtE10CaMSlpCdy4HAFy/NLj1meYltpGQAtooQB3eK5cnh3Nxs5tT29ktoSZV1F9coShDGNXa1KSccfHTCuLiVt5K2CaiUHDvNFbhGf5xaSvWk363bsxespIrVjKzLTOzIspT3D0xIY3L09cTGNcwykKbEEYUxjUn7QGzSuUUp6sPjoMRFSSnoqZeWVZG/azWc5hczPLWR5QSmqrqpoRI8OXDyoC8NSExiWmkBS21i/w22WLEEY44dgpaQ7VkJx7vGlpAMvj4hS0lOprFJWFJQeSQgLN+3mcEUVLaKE4WkJTLuwL2P7JDEsNSEsy1qb2rMEYUy4Hdp3/DjBcaWk6S4JDLkqIkpJQ6GqbCzcz+c5hXyeU8T83EL2eHMPBnSN5/tn92BsnyRG9ezYLCehRQL7v2JMOBzcDWvfdctV534IlW6p50gsJa2NnXvLmJ9T5CWFQrZ6g8rJCa2YOKQr5/RJYkzvJDrFW5dRJLAEYUx92V/obmizKsvd4KaqAtqlwJk3Qs9zXTKIgFLS2th3qIIvNxTxuZcU1u7YC0D7VjGM6Z3IrecnMbZPEj0SWzfK+x2Yk7MEYczp2Lvd3fpy1WzY/LkbP+jQE0bf5u5h0H1Ek0oI5ZVVLM0r4bP1roWwNK+EiioltkUUZ6Z35MrhyYztk8Sg7u1sdnITYAnCmNoqzffuiZwFWxYACkn94Nx73H2Ruw5tMklBVVmzfe+RLqMvNxZz4HAlInBGcntuGteLsX2SGNGjg5WdNkGWIIwJRfFGlxBWzYaCRW5blyEw/meupdB5gL/x1aP83QeYn1N0pNqocJ8bP+mV1IarR6RwTp8kRvdKpH3ryCqrNbVnCcKYEylc7xLCqtmw/Wu3rVsGXPigSwqJvX0Nr75UT1D7zGslVN8uM6ltLGP7JDGmTxLn9EkiOaHxz7429csShDHVVN0ktVWzXWth5yq3PWUUXPwIDLzClZ5GuMAJap/nFLJi69EJamf3SuT7o9MZ2yeJfl3a2sByM2cJwjRvqrBt2dHuo6IcQKDHGLjkdzDgcmif7HeUpyVwgtrnOYVkbz46QW1EWgfuvLAf5/RJtAlq5jiWIEzzU1XlxhFWz3aDzSWbQaJdKerZt7qkEN/F7yjrLHCC2mc5hXyRW2QT1EydhPVfh4hMBB4HooEZqvpojf03AL8HCrxNT6nqDG/f9cD93vZHVPWFcMZqmriqSsj78mj10Z4CiIpxi9uN+wn0vxTaJPodZZ1VT1D7LKeQ+TZBzdSTsCUIEYkGpgMTgHxgoYhkqeqqGoe+qqq31zi3I/AgkAkosMg7d3e44jVNUGWFm5uwarabwLZvB0THQp+L4MIHoN9EaJXgd5R1Uj1BzSWEIpugZsIinC2IUUCOqm4AEJFXgMlAzQQRzDeAD1S12Dv3A2Ai8HKYYjVNRcVh2PiJ6z5a8zYcKHKL3vWd4CqP+l4ckctaHK5wE9Sq5yPYBDXTEMKZIJKBvIDn+cBZQY67WkTGAeuAu1Q17wTnHjdSKCI3ATcBpKWl1VPYJuKUl8GGua6lsPYdKCuFlvHQf6KbuNbnImjZ2u8oa6WqSlm74/gJalECQ22Cmmkgfo9QvQm8rKqHROTHwAvABaGerKrPAc8BZGZmanhCNI3S4QOQ84EbU1j3bzi8D+LaQ//LXEuh13iIiazbTebvPnDMyqc2Qc34LZwJogBIDXiewtHBaABUtSjg6QzgdwHnjq9x7rx6j9BElkN7Yd17rqWQMwfKD7hbag65GgZNgvRx0KKl31GG7FQT1M7xfrrbBDXjk3AmiIVAXxHpifvCnwpcG3iAiHRT1eob6E4CVnuP3wP+R0Q6eM8vBn4WxlhNY3WwxC2bvToLcj6EykPQtgtkXOtaCmljINrvhnBoTjVB7brR6ZxjE9RMIxK2T5aqVojI7bgv+2jgeVVdKSIPA9mqmgXcISKTgAqgGLjBO7dYRH6FSzIAD1cPWJtmYH8RrH3btRQ2fOzuu9wuBc78oRtTSD0LoiJjQtf+QxW8ujCPOat3BJ2gNrZvImek2AQ10ziJatPous/MzNTs7Gy/wzB1tXcHrHnTjSls+gy00i1rMXASDLoSkiNr2ezd+w/z1/mb+Ov8TZQeLGdA13jXbdQ3iVHpNkHNNB4iskhVM4Pts3+lxj+lBUfvpbDlC0AhsS+MvcuNKXQ9I6KSAsCOPWXM+HQDL365hQOHK7l4UBduPb8PGakJfodmTK1ZgjANa/emo7OZ870exM6DYfx9bkyh04CISwoAW4oO8MwnuczKzqdSlUnDunPzeb3p3zXy5lwYU80ShAm/whxv3aPZbmE8gG7D3GzmgZMhqY+/8Z2GNdv38Od5uby5bCstoqL4dmYKPx7Xm7TEyJp3YUwwliBM/VOFXWu8eylkwc6VbnvKmTDhV677KMKXzV68ZTdPz81lzuodtGkZzY/O7cWPxvakc7vImnthzMlYgjD1Q9XdVGdV9bLZ6wGBtNEw8bcw8HJon+J3lKdFVfk8p4in5+UwP7eIhNYx3HVRP64f04OE1pEz/8KYUFmCMHWnCgWLYdUbbkxh9ya3bHb6WDj7ZhhwRUQvm12tqkr5YPUOnp6bw7L8UjrHx3L/ZQO5ZlSaVSOZJs3+dZvaK1gEy2e51sKefG/Z7PPg3HvcUhcRvGx2oIrKKt78eitPz81l/c59pHVszf98cyhXj0wmtoWtf2SaPksQJnSF6+GDB90ktuhY6HMhXHC/WxSvVYdTnx8hysormbUon2c/ySWv+CD9u8Tz+NQMLhvajRY2oc00I5YgzKntL4R5j0L2827p7AsfgDNvhLh2fkdWr/YdquDFBZuZ8dlGdu09REZqAg9ePpgLBnQmypbQNs2QJQhzYuUHYcGf4dPH3MJ4mf8B5/0U2nbyO7J6tXv/Yf5v/iZe8GY9j+2TxONTMxjdK9HWRDLNmiUIc7yqKlg+Ez78lRtj6H8ZXPQQdOrnd2T1anupm/X80ldu1vM3Bnfh1vF9GGazno0BLEGYmjZ+Au/f7ya0dR8OVz3rqpKakE2F+3n2k1xeW1RApSqTh3Xn5vG96dfFZj0bE8gShHF2rYUPHnA332mfClfNcPdZiJBVU0Oxepub9fzW11tpER3Fd850s55TO9qsZ2OCsQTR3O3bCfN+A4tegJZt4aJfwlk3R9zd2E5m0ebdPD03hw/X7KRNy2huPLcXP7RZz8ackiWI5urwAVgwHT77E1SUwZk/cgPQTWQOg6ryWU4h0+fmsGBDMQmtY7h7Qj+uH51ut+w0JkSWIJqbqkr4+lU3AL13Kwy43LUaInjBvEBVVcr7q3bw9Lwcvs4vpUs7m/VsTF3ZJ6Y5yZ0LH/wCti+H5JHwreehx2i/o6oX5ZVVvLlsK0/PyyVn5z56JLbmN1cN5aoRNuvZmLqyBNEc7FjlBqBzPoCENJcYBl8VkfddqKmsvJJ/ZufxzMcbKCg5yICuNuvZmPpiCaIp27sd5v4PLPk7xMbDxY/AqJugRazfkZ22vWXlvPjlFmZ8upHCfYcYnpbAw5PdrGeb3GZM/bAE0RQd3g/zn4LPH4fKw64qadxPoHVHvyM7bcX7D/PXzzfy1/mb2FNWwbl9k7h1/HDO7tXREoMx9cwSRFNSVQlLX4KPHoF9290tPC96CDr28juy07at9CB/+WQjL3+1hYPllUwc3JVbz+/NGSkJfodmTJNlCaKpyPkQ3v+Fu3tbypnwnb9B2ll+R3XaNhbu59mPc3ltcT5VCpMzunPLeb3pa7OejQk7SxCRbvsKV5mU+5G7jee3X3Athwjvblm9bQ9Pz8vlbW/W89Qz07hpXC+b9WxMAzplghCRK4C3VbWqAeIxodqzDeY+AktehLj28I3fwJk/jPgB6EWbi5k+N5eP1uykbWwLbhznzXqOt1nPxjS0UFoQU4A/ichrwPOquibUFxeRicDjQDQwQ1UfPcFxVwOzgDNVNVtE0oHVwFrvkAWqenOo79ukHdoH85+A+U9CVQWMvg3G3RvRN+xRVT5d72Y9f7mxmA6tY7hnQj+us1nPxvjqlAlCVb8nIu2Aa4C/iogC/we8rKp7T3SeiEQD04EJQD6wUESyVHVVjePigWnAlzVeIldVM2rzyzRplRWw9B/w0a9h/063kN6FD7hupQjlZj1vZ/rcXJYXlNK1XRy/uHwQ14xKpXVL6/00xm8hfQpVdY+IzAJaAXcC3wR+IiJPqOqTJzhtFJCjqhsAROQVYDKwqsZxvwJ+C/yk9uE3A6qw/gM3zrBrDaSeDde8DCmZfkdWZ+WVVcxeupU/z8shd9d+eiS25tGrhvJNm/VsTKMSyhjEJOAHQB/gb8AoVd0pIq1xX/YnShDJQF7A83zgmLIaERkBpKrq2yJSM0H0FJElwB7gflX9NEhsNwE3AaSlpZ3qV4k8275292bY+LErVf3O32HgFRE7AF1WXsnM7DyeDZj1/MQ1w7l0SFeb9WxMIxRKC+Jq4I+q+kngRlU9ICI/rOsbi0gU8BhwQ5Dd24A0VS0SkZHAGyIyWFX31IjhOeA5gMzMTK1rLI1OaYGby7DsZTe2cMnvYOQPoEVLvyOrkz1l5fxjwWae/2wjhfsOMyItgV9dOZjz+9usZ2Mas1ASxEO4L2wARKQV0EVVN6nqhyc5rwBIDXie4m2rFg8MAeZ5XxJdgSwRmaSq2cAhAFVdJCK5QD8gO4R4I9ehvW757S+mg1bBOXfA2LuhVYLfkdVJ0b5D/N/nm3jhi03s9WY933Z+H87qabOejYkEoSSIfwJjAp5XetvOPMV5C4G+ItITlximAtdW71TVUiCp+rmIzAPu9aqYOgHFqlopIr2AvsCGEGKNTJUVsPgFd+Oe/btg6Lfhgl9Ahx5+R1YnW0sO8pdPN/DyV1s4VFHlZj2P78PQlPZ+h2aMqYVQEkQLVT1c/URVD4vIKfs6VLVCRG4H3sOVuT6vqitF5GEgW1WzTnL6OOBhESkHqoCbVbU4hFgjiyqse88NQBeugx7nwLWvuqW4I9DGwv08My+Xfy1xs56vzEjmlvG96NPZZj0bE4lCSRC7vG6fLAARmQwUhvLiqvoO8E6NbQ+c4NjxAY9fA14L5T0i1tYlbmmMTZ9CYh+Y+hL0vzQiB6BXbi3l6Xm5vLt8Gy2io7hmVBo3nmuzno2JdKEkiJuBF0XkKUBwlUnXhTWqpqwkDz76lburW+tEuPR/YeQNEB15E8KyNxUzfW4Oc9fuom1sC24a15v/GJtus56NaSJCmSiXC5wtIm295/vCHlVTVFYKn/0RvnjatRLG3g1j73TLZESY+bmF/GnOer7aWEzHNi259+J+fH90Ou1bRV6SM8acWEgT5UTkMmAwEFddfaKqD4cxrqajshwW/dUNQB8ogjOmwgX3Q0LqKU9tjLI3FfO9GV/SOT6OBy4fxFSb9WxMkxXKRLlngNbA+cAM4FvAV2GOK/Kpwtp33K0+i3Ig/Vx3R7fuGX5HVmd7ysqZ9spSkju04p07ziU+zloMxjRlofzpN0ZVzxCRr1X1lyLyB+DdcAcW0QoWuQHozZ9DUj+45lXo942IHIAO9Is3VrB9TxkzfzzakoMxzUAoCaLM++8BEekOFAHdwhdSBNu9GT58GFbMgjad4LLHYMT1EB35XTCvL8ln9tKt3HVRP0b2iNyVY40xoQvlm+tNEUkAfg8sBhT4SziDijgHS+DTP8CXz4BEu/s/nzMNYptG/f+WogP84o2VZPbowG3n9/Y7HGNMAzlpgvDWS/pQVUuA10TkLSDOmwVtKg5D9vPw8aMuSWRcC+f/N7RP9juyelNRWcWdry5BgD9OybBF9YxpRk6aIFS1SkSmA8O954fw1khq1lRhdRbMeQiKN0DP89wAdLcz/I6s3j3xUQ6Lt5Tw+NQMm/hmTDMTShfTh94d3/6lqk1nxdS6ys+G9/4b8hZApwHw3VnQ56KIH4AOZuGmYp76aD1XDU9mckbTaRUZY0ITSoL4MXA3UCEiZbjZ1Kqq7cIaWWNTvNENQK/8F7TpDFc8DhnfaxID0MGUHiznzleWktKhNb+cPNjvcIwxPghlJnXTGGmtqwPF3gD0s245jPN+CmPugNi2fkcWNqp6pKT1nzdbSasxzVUoE+XGBdte8wZCTU7FIVg4Az7+nVsmY/j33AB0u6Zf4fvG0gKylm3l7gn9GJFmJa3GNFeh9I8E3go0Dnev6UXABWGJyG+qsOoNNwC9exP0vhAmPAxdh/gcWMOoLmk9M70Dt53fx+9wjDE+CqWL6YrA5yKSCvwpXAH5asuX8P5/Q/5C6DwYvveaG4BuJo6UtIoraY2OanoD78aY0NVlhDUfGFjfgfiqKNe1GFZnQduuMOkpN6chKtrvyBpUdUnrE9cMJ6WDlbQa09yFMgbxJG72NEAUkIGbUd00FObA02dDdEsY/3MYczu0bON3VA3uSEnriGQmDevudzjGmEYglBZEdsDjCuBlVf08TPE0vKQ+boxhyFUQ39XvaHwRWNL68OTmMdZijDm1UBLELKBMVSsBRCRaRFqr6oHwhtaARt/qdwS+qVnS2ja2ac7rMMbUXigL63wItAp43gqYE55wTEN7fYkraZ12YV8raTXGHCOUBBEXeJtR77GNYDYBW4oO8MBsK2k1xgQXSoLYLyIjqp+IyEjgYPhCMg2hvLKKaVbSaow5iVA6nO8E/ikiW3HrMHUFpoQzKBN+T364niVW0mqMOYlQJsotFJEBQH9v01pVLQ9vWCacvtpYzFNzc6yk1RhzUqfsYhKR24A2qrpCVVcAbUWk+Zb9RLjSg+Xc9aqVtBpjTi2UMYgbvTvKAaCqu4EbQ3lxEZkoImtFJEdE7jvJcVeLiIpIZsC2n3nnrRWRb4TyfubkVJX7vZLWx6dmWEmrMeakQvmGiBYRqb5ZkIhEAy1PdZJ33HRgAm55joUikqWqq2ocFw9MA74M2DYImAoMBroDc0SkX/VcDFM3/1pcwJvLtnLPhH4Mt5JWY8wphNKC+DfwqohcKCIXAi8D74Zw3iggR1U3qOph4BVgcpDjfgX8FigL2DYZeEVVD6nqRiDHez1TR5uL9vPA7BWMSu/IrVbSaowJQSgJ4qfAR8DN3s9yjp04dyLJQF7A83xv2xFe+Wyqqr5d23O9828SkWwRyd61a1cIITVP5ZVV3PnqUqKihD9OtZJWY0xoTpkgVLUK1/2zCfdX/AXA6tN9YxGJAh4D7qnra6jqc6qaqaqZnTp1Ot2Qmqzqktb/+eZQkhNCye3GGHOSMQgR6Qdc4/0UAq8CqOr5Ib52AZAa8DzF21YtHhgCzBMRcPMrskRkUgjnmhBVl7RePSKFK6yk1RhTCydrQazBtRYuV9WxqvokUJtB4oVAXxHpKSItcYPOWdU7VbVUVZNUNV1V04EFwCRVzfaOmyoisSLSE+gLfFWr38wcKWlN7diaX04e7Hc4xpgIc7IEcRWwDZgrIn/xBqhD7rxW1QrgduA9XJfUTFVdKSIPe62Ek527EpgJrMINkt9mFUy1E1jS+qcpVtJqjKk98apXT3yASBtcVdE1uBbF34DXVfX98IcXuszMTM3Ozj71gc3Ea4vyueefy7j34n7cfkFfv8MxxjRSIrJIVTOD7QtlkHq/qr7k3Zs6BViCq2wyjVRgSest462k1RhTN6GUuR6hqru9yqELwxWQOT3llVVMe8VKWo0xp886ppuYJz5cz9K8Ep68ZriVtBpjTkutWhCmcftqYzHTraTVGFNPLEE0EVbSaoypb9bF1ASoKv/9+nK27ylj1s2jraTVGFMvrAXRBLy2uIC3vt7GXRf1tVVajTH1xhJEhNtUuJ8HZ69gVE8raTXG1C9LEBGsepXW6Cjhj1OspNUYU7+sszqCVZe0PnWtlbQaY+qftSAiVHVJ67dGpnD5GVbSaoypf5YgIlBgSetDk6yk1RgTHtbFFGGqS1p37Clj1i1jrKTVGBM21oKIMEdKWif0IyM1we9wjDFNmCWICFJd0npWz47cfF5vv8MxxjRxliAiRHllFdOspNUY04CsAztCPD5nPcu8ktbuVtJqjGkA1oKIAF9uKGL6PCtpNcY0LEsQjVzpAVfSmmYlrcaYBmZdTI2YqvLzN5azc+8hK2k1xjQ4a0E0YrMW5fO2lbQaY3xiCaKR2lS4nwezVlpJqzHGN5YgGqHyyiqmvbKEFlbSaozxkXVqN0KPz1nPsvxSpl87wkpajTG+CWsLQkQmishaEckRkfuC7L9ZRJaLyFIR+UxEBnnb00XkoLd9qYg8E844G5PqktZvj0zhsjO6+R2OMaYZC1sLQkSigenABCAfWCgiWaq6KuCwl1T1Ge/4ScBjwERvX66qZoQrvsaouqS1h5W0GmMagXC2IEYBOaq6QVUPA68AkwMPUNU9AU/bABrGeBq1wJLWx6cOp42VtBpjfBbOBJEM5AU8z/e2HUNEbhORXOB3wB0Bu3qKyBIR+VhEzg1jnI1CYEnrMCtpNcY0Ar5XManqdFXtDfwUuN/bvA1IU9XhwN3ASyLSrua5InKTiGSLSPauXbsaLuh6ZiWtxpjGKJwJogBIDXie4m07kVeAKwFU9ZCqFnmPFwG5QL+aJ6jqc6qaqaqZnTp1qq+4G1R1SWtMdJSVtBpjGpVwJoiFQF8R6SkiLYGpQFbgASLSN+DpZcB6b3snb5AbEekF9AU2hDFW3/xpzjqW5Zfym6uGWkmrMaZRCdtIqKpWiMjtwHtANPC8qq4UkYeBbFXNAm4XkYuAcmA3cL13+jjgYREpB6qAm1W1OFyx+mXBhiKenpfLdzJTuHSolbQaYxoXUW0ahUOZmZmanZ3tdxghKz1QzsTHPyG2RRRv33GuVS0ZY3whIotUNTPYPvtW8oGq8vPXl7Nr7yFeu2WMJQdjTKPkexVTc/TPRfm8vdxKWo0xjZsliAa2sXA/D1lJqzEmAliCaEDllVXcaSWtxpgIYZ3fDeiPH7iS1qe/a6u0GmMaP2tBNJAFG4r488dW0mqMiRyWIBpA9Sqt6YltePAKW6XVGBMZrIspzKyk1RgTqawFEWbVJa13X2wlrcaYyGIJIoyqS1rP7tWRH4+zklZjTGSxBBEmVtJqjIl01iEeJtUlrX/+7gi6tbeSVmNM5LEWRBh8ketKWqdkpnKJlbQaYyKUJYh6VnLgMHfPdCWtD1wxyO9wjDGmzqyLqR4FlrT+61YraTUmEpSXl5Ofn09ZWZnfoYRVXFwcKSkpxMTEhHyOfYPVo39m5/PO8u3818T+nJGS4Hc4xpgQ5OfnEx8fT3p6OiJNs5hEVSkqKiI/P5+ePXuGfJ51MdWTjYX7eehNK2k1JtKUlZWRmJjYZJMDgIiQmJhY61aSJYh6cLiiimlW0mpMxGrKyaFaXX5H62KqB3+cs46vraTVGNPEWAviNM3PLeQZK2k1xtRRSUkJTz/9dK3Pu/TSSykpKan/gAJYgjgNJQcOc/ery6yk1RhTZydKEBUVFSc975133iEhISFMUTnWxVRH1SWthfuspNWYpuKXb65k1dY99fqag7q3O+ky//fddx+5ublkZGQQExNDXFwcHTp0YM2aNaxbt44rr7ySvLw8ysrKmDZtGjfddBMA6enpZGdns2/fPi655BLGjh3L/PnzSU5OZvbs2bRqdfrd3daCqKPqktZ7LraSVmNM3T366KP07t2bpUuX8vvf/57Fixfz+OOPs27dOgCef/55Fi1aRHZ2Nk888QRFRUXHvcb69eu57bbbWLlyJQkJCbz22mv1Epv92VsH1SWto3sl8uNxvfwOxxhTTxrDDb1GjRp1zFyFJ554gtdffx2AvLw81q9fT2Ji4jHn9OzZk4yMDABGjhzJpk2b6iUWSxC1FFjS+tiUYURZSasxph61adPmyON58+YxZ84cvvjiC1q3bs348eODzmWIjY098jg6OpqDBw/WSyxh7WISkYkislZEckTkviD7bxaR5SKyVEQ+E5FBAft+5p23VkS+Ec44a6O6pPW3Vw+1klZjzGmLj49n7969QfeVlpbSoUMHWrduzZo1a1iwYEGDxha2FoSIRAPTgQlAPrBQRLJUdVXAYS+p6jPe8ZOAx4CJXqKYCgwGugNzRKSfqlaGK95QVJe0Tj0zlYlDrKTVGHP6EhMTOeeccxgyZAitWrWiS5cuR/ZNnDiRZ555hoEDB9K/f3/OPvvsBo0tnF1Mo4AcVd0AICKvAJOBIwlCVQPLBdoA6j2eDLyiqoeAjSKS473eF2GM96SqS1p7WkmrMaaevfTSS0G3x8bG8u677wbdVz3OkJSUxIoVK45sv/fee+strnB2MSUDeQHP871txxCR20QkF/gdcEctz71JRLJFJHvXrl31FnhNqsrP/rWcov2HeHzqcFq3tKEbY0zT53uZq6pOV9XewE+B+2t57nOqmqmqmZ06dQpPgMDM7DzeXeFKWoemtA/b+xhjTGMSzgRRAKQGPE/xtp3IK8CVdTw3bDbs2sdDWasY3SuRm861klZjTPMRzgSxEOgrIj1FpCVu0Dkr8AAR6Rvw9DJgvfc4C5gqIrEi0hPoC3wVxliDciWtS2nZwkpajTHNT9g601W1QkRuB94DooHnVXWliDwMZKtqFnC7iFwElAO7geu9c1eKyEzcgHYFcJsfFUyPfbCO5QWlPPM9W6XVGNP8hHW0VVXfAd6pse2BgMfTTnLur4Ffhy+6k5ufW8izn1hJqzGm+fJ9kLox2r3fSlqNMY1T27ZtG+y9LEHUUL1Kq5W0GmOaO/v2q6G6pPW+SwZYSasxzc2798H25fX7ml2HwiWPnnD3fffdR2pqKrfddhsADz30EC1atGDu3Lns3r2b8vJyHnnkESZPnly/cYXAWhABqktax/S2klZjTMOYMmUKM2fOPPJ85syZXH/99bz++ussXryYuXPncs8996CqJ3mV8LAWhKe6pDU2JorHvpNhJa3GNEcn+Us/XIYPH87OnTvZunUru3btokOHDnTt2pW77rqLTz75hKioKAoKCtixYwddu3Zt0NgsQXiOlrSOpGv7OL/DMcY0I9/+9reZNWsW27dvZ8qUKbz44ovs2rWLRYsWERMTQ3p6etBlvsPNEgRHS1qvGZXKxCENm6GNMWbKlCnceOONFBYW8vHHHzNz5kw6d+5MTEwMc+fOZfPmzb7E1ewTxJGS1qQ2/OJyK2k1xjS8wYMHs3fvXpKTk+nWrRvf/e53ueKKKxg6dCiZmZkMGDDAl7iafYKoVGVIcnvuvKivlbQaY3yzfPnR6qmkpCS++CL43Q327dvXUCFZgkhqG8uM6zP9DsMYYxodK3M1xhgTlCUIY0yz58ccg4ZWl9/REoQxplmLi4ujqKioSScJVaWoqIi4uNqV8Df7MQhjTPOWkpJCfn4+4bxtcWMQFxdHSkpKrc6xBGGMadZiYmLo2bOn32E0StbFZIwxJihLEMYYY4KyBGGMMSYoaSoj9yKyCzidBUuSgMJ6Cqc+WVy1Y3HVjsVVO00xrh6q2inYjiaTIE6XiGSraqObUm1x1Y7FVTsWV+00t7isi8kYY0xQliCMMcYEZQniqOf8DuAELK7asbhqx+KqnWYVl41BGGOMCcpaEMYYY4KyBGGMMSaoZpUgRGSiiKwVkRwRuS/I/lgRedXb/6WIpDeSuG4QkV0istT7+VEDxfW8iOwUkRUn2C8i8oQX99ciMqKRxDVeREoDrtcDDRRXqojMFZFVIrJSRKYFOabBr1mIcTX4NROROBH5SkSWeXH9MsgxDf6ZDDEuXz6T3ntHi8gSEXkryL76vV6q2ix+gGggF+gFtASWAYNqHHMr8Iz3eCrwaiOJ6wbgKR+u2ThgBLDiBPsvBd4FBDgb+LKRxDUeeMuH69UNGOE9jgfWBfl/2eDXLMS4Gvyaedegrfc4BvgSOLvGMX58JkOJy5fPpPfedwMvBfv/Vd/Xqzm1IEYBOaq6QVUPA68Ak2scMxl4wXs8C7hQRKQRxOULVf0EKD7JIZOBv6mzAEgQkW6NIC5fqOo2VV3sPd4LrAaSaxzW4NcsxLganHcNqm+wHOP91KyaafDPZIhx+UJEUoDLgBknOKRer1dzShDJQF7A83yO/5AcOUZVK4BSILERxAVwtdclMUtEUsMcU6hCjd0Po70ugndFZHBDv7nXtB+O++szkK/X7CRxgQ/XzOsuWQrsBD5Q1RNerwb8TIYSF/jzmfwT8F9A1Qn21+v1ak4JIpK9CaSr6hnABxz9C8EEtxi3vsww4EngjYZ8cxFpC7wG3KmqexryvU/mFHH5cs1UtVJVM4AUYJSIDGmI9z2VEOJq8M+kiFwO7FTVReF+r2rNKUEUAIFZPsXbFvQYEWkBtAeK/I5LVYtU9ZD3dAYwMswxhSqUa9rgVHVPdReBqr4DxIhIUkO8t4jE4L6EX1TVfwU5xJdrdqq4/Lxm3nuWAHOBiTV2+fGZPGVcPn0mzwEmicgmXFf0BSLyjxrH1Ov1ak4JYiHQV0R6ikhL3ABOVo1jsoDrvcffAj5Sb7THz7hq9FFPwvUhNwZZwHVeZc7ZQKmqbvM7KBHpWt3vKiKjcP/Ow/6l4r3n/wNWq+pjJziswa9ZKHH5cc1EpJOIJHiPWwETgDU1Dmvwz2QocfnxmVTVn6lqiqqm474nPlLV79U4rF6vV7O55aiqVojI7cB7uMqh51V1pYg8DGSrahbuQ/R3EcnBDYJObSRx3SEik4AKL64bwh0XgIi8jKtuSRKRfOBB3IAdqvoM8A6uKicHOAD8oJHE9S3gFhGpAA4CUxsg0YP7C+/7wHKv/xrg50BaQGx+XLNQ4vLjmnUDXhCRaFxCmqmqb/n9mQwxLl8+k8GE83rZUhvGGGOCak5dTMYYY2rBEoQxxpigLEEYY4wJyhKEMcaYoCxBGGOMCcoShDG1ICKVASt4LpUgq++exmunywlWqDXGD81mHoQx9eSgtwSDMU2etSCMqQcisklEficiy8XdS6CPtz1dRD7yFnX7UETSvO1dROR1b3G8ZSIyxnupaBH5i7j7ELzvzeQ1xheWIIypnVY1upimBOwrVdWhwFO4VTfBLXz3greo24vAE972J4CPvcXxRgArve19gemqOhgoAa4O629jzEnYTGpjakFE9qlq2yDbNwEXqOoGb2G87aqaKCKFQDdVLfe2b1PVJBHZBaQELPhWvRT3B6ra13v+UyBGVR9pgF/NmONYC8KY+qMneFwbhwIeV2LjhMZHliCMqT9TAv77hfd4PkcXTPsu8Kn3+EPgFjhyc5r2DRWkMaGyv06MqZ1WASuiAvxbVatLXTuIyNe4VsA13rb/BP5PRH4C7OLo6q3TgOdE5Ie4lsItgO9LpRsTyMYgjKkH3hhEpqoW+h2LMfXFupiMMcYEZS0IY4wxQVkLwhhjTFCWIIwxxgRlCcIYY0xQliCMMcYEZQnCGGNMUP8f75RBvgpiplwAAAAASUVORK5CYII=\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    }
  ]
}