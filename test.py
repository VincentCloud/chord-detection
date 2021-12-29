from __future__ import print_function, division

import cv2
import time
from PIL import Image
from torchvision.transforms import transforms
from utils.img_utils import pad_to_square
import numpy as np
import torch

from transforms.rescale import Rescale
from utils.utils import AverageMeter, accuracy, init_model_and_dataset
from utils.img_utils import compute_gradient, save_img, single_frame_operation
from sklearn.cluster import KMeans


def test(val_loader, model, device, save_imgs=False, show=False):
    batch_time = AverageMeter()

    eval_fingers_recall = AverageMeter()
    eval_fingers_precision = AverageMeter()

    eval_frets_recall = AverageMeter()
    eval_frets_precision = AverageMeter()

    eval_strings_recall = AverageMeter()
    eval_strings_precision = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for data_idx, data in enumerate(val_loader):
        input = data['image'].float().to(device)
        print(f'input dimensions: {input.shape}')
        target = data['fingers'].float().to(device)
        frets = data['frets'].float().to(device)
        strings = data['strings'].float().to(device)
        target_coord = data['finger_coord']
        frets_coord = data['fret_coord']
        strings_coord = data['string_coord']
        img_number = data['img_number']

        # compute output
        output = model(input)
        output1 = output[0].split(input.shape[0], dim=0)
        output2 = output[1].split(input.shape[0], dim=0)
        output3 = output[2].split(input.shape[0], dim=0)

        if show:
            import matplotlib.pyplot as plt
            import torchvision.transforms as transforms
            fig, ax = plt.subplots(1, 5)
            ax[0].imshow(target[0][0].cpu(), cmap='gray')
            ax[1].imshow(output1[-1][0][0].cpu().detach(), cmap='gray')
            ax[2].imshow(output2[-1][0][0].cpu().detach(), cmap='gray')
            ax[3].imshow(output3[-1][0][0].cpu().detach(), cmap='gray')
            ax[4].imshow(transforms.ToPILImage()(input.cpu()[0]))
            plt.show()

        # measure accuracy
        accuracy(output=output1[-1].data, target=target,
                 global_precision=eval_fingers_precision, global_recall=eval_fingers_recall, fingers=target_coord,
                 min_dist=10)

        accuracy(output=output2[-1].data, target=frets,
                 global_precision=eval_frets_precision, global_recall=eval_frets_recall,
                 fingers=frets_coord.unsqueeze(0), min_dist=5)

        accuracy(output=output3[-1].data, target=strings,
                 global_precision=eval_strings_precision, global_recall=eval_strings_recall,
                 fingers=strings_coord.unsqueeze(0), min_dist=5)

        if save_imgs:
            save_img(input.cpu().detach()[0], output1[-1][0][0].cpu().detach().numpy(), 10,
                     'image{num}_fingers'.format(num=data['img_number'][0]))
            save_img(input.cpu().detach()[0], output2[-1][0][0].cpu().detach().numpy(), 5,
                     'image{num}_frets'.format(num=data['img_number'][0]))
            save_img(input.cpu().detach()[0], output3[-1][0][0].cpu().detach().numpy(), 5,
                     'image{num}_strings'.format(num=data['img_number'][0]))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        break

    # print('FINGERS: \t'
    #       'Recall(%): {top1:.3f}\t'
    #       'Precision(%): {top2:.3f}\n'
    #       'FRETS:   \t'
    #       'Recall(%): {top6:.3f}\t'
    #       'Precision(%): {top7:.3f}\n'
    #       'STRINGS: \t'
    #       'Recall(%): {top11:.3f}\t'
    #       'Precision(%): {top12:.3f}\n'
    #     .format(top1=eval_fingers_recall.avg * 100, top2=eval_fingers_precision.avg * 100,
    #     top6=eval_frets_recall.avg * 100, top7=eval_frets_precision.avg * 100,
    #     top11=eval_strings_recall.avg * 100, top12=eval_strings_precision.avg * 100))
    #
    # return eval_fingers_recall.avg, eval_frets_recall.avg, eval_strings_recall.avg, eval_fingers_precision.avg, \
    #        eval_frets_precision.avg, eval_strings_precision.avg


def test_from_camera(model, device, ckpt):
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print('Cannot open camera')
    else:
        while True:
            ret, frame = cap.read()

            if not ret:
                print('Cannot print frame, exiting')
                break
            annotated_frame = evaluate_single_frame(frame, model, device, ckpt)

            cv2.imshow('frame', annotated_frame)
            if cv2.waitKey(1) == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()


def evaluate_single_frame(frame, model, device, ckpt):
    checkpoint = torch.load(ckpt, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    with torch.no_grad():
        model.eval()
        hand_cropped = single_frame_operation(frame)
        if hand_cropped.any():
            # rescale
            padded_image = pad_to_square(hand_cropped)
            rescaled_image = cv2.resize(padded_image, (300, 300))
            transform = transforms.ToTensor()
            image_tensor = transform(cv2.cvtColor(rescaled_image, cv2.COLOR_BGR2RGB))
            image_tensor_unsqueezed = torch.unsqueeze(image_tensor, 0)
            # dataloader = torch.utils.data.DataLoader(image_tensor, batch_size=1, shuffle=False)

            # for data_idx, data in enumerate(dataloader):
            image_data = image_tensor_unsqueezed.float().to(device)

            print(f'image tensor dimensions: {image_data.shape}')
            output = model(image_data)
            output1 = output[0].split(image_data.shape[0], dim=0)
            output2 = output[1].split(image_data.shape[0], dim=0)
            output3 = output[2].split(image_data.shape[0], dim=0)
            print(f'output shape: {output[0].shape} {output[1].shape} {output[2].shape}')
            fingers = output1[-2][0][0].cpu().numpy()
            # print(f'finger values: {sorted(list(np.unique(fingers)), reverse=True)}')
            i, j = np.unravel_index(np.argmax(fingers), fingers.shape)
            fingers[i, j] = 0.9
            cv2.imshow('original', rescaled_image)
            # plot_pixels(output1[-2][0][0].cpu().numpy())
            reformed_heatmap = get_brightest_pixels(output1[-2][0][0].cpu().numpy())
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 4)
            ax[0].imshow(output1[-2][0][0].cpu().numpy(), cmap='gray')
            ax[1].imshow(output2[-2][0][0].cpu().numpy(), cmap='gray')
            ax[2].imshow(output3[-2][0][0].cpu().numpy(), cmap='gray')
            ax[3].imshow(cv2.cvtColor(rescaled_image, cv2.COLOR_BGR2RGB), cmap='gray')
            plt.show()
            cv2.waitKey(0)
            # cv2.imshow('output1', get_brightest_pixels(output1[-2][0][0].cpu().numpy()))
            # cv2.imshow('output2', output2[-2][0][0].cpu().numpy())
            # cv2.imshow('output3', output3[-2][0][0].cpu().numpy())
            # cv2.circle(rescaled_image, (j, i), 0, (255, 0, 0), 2)

            return rescaled_image
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(1, 2)
            # ax[0].imshow(fingers, cmap='gray')
            # ax[1].imshow(cv2.cvtColor(rescaled_image, cv2.COLOR_BGR2RGB), cmap='gray')
            # plt.show()

            # print(f'output dimension: {output.size()}')
        else:
            print('There is no hand detected')


def read_image(model, device, image_path, ckpt):
    image = cv2.imread(image_path)
    evaluate_single_frame(image, model, device, ckpt)


def get_brightest_pixels(heatmap):
    # flat_indices = np.argpartition(heatmap.ravel(), len(heatmap.ravel()) - n - 1)[:-n]
    # row_indices, col_indices = np.unravel_index(flat_indices, heatmap.shape)
    # return row_indices, col_indices
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    # k = 2
    # retval, labels, centers = cv2.kmeans(heatmap, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    #
    # centers = np.uint8(centers)
    # segmented_data = centers[labels.flatten()]
    #
    # segmented_image = segmented_data.reshape(heatmap.shape)
    #
    # # cv2.imshow('segmented image', segmented_image)
    # return segmented_image
    kmeans = KMeans(n_clusters=2, random_state=0).fit(heatmap.reshape(-1, 1))
    reformed_heatmap = kmeans.predict(heatmap.reshape(-1, 1))
    return np.reshape(reformed_heatmap, heatmap.shape)

def plot_pixels(heatmap):
    pixel_values = heatmap.flatten()
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hlines(1, 1, 20)
    plt.eventplot(pixel_values, orientation='horizontal', colors='b')
    plt.xlim(-2 , 2)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    num_workers = 0
    directory = 'data/'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, train_dataset, val_dataset, _, _ = init_model_and_dataset(directory, device)

    read_image(model, device, '/Users/vincenthuang/Development/Study/MUMT502/chord-detection/data/2/image210.jpg', '/Users/vincenthuang/Development/Study/MUMT502/chord-detection/checkpoints/best_ckpt/MTL_hourglass.pth')
    # test_from_camera(model, device, '/Users/vincenthuang/Development/Study/MUMT502/chord-detection/checkpoints'
    #                                 '/best_ckpt/MTL_hourglass.pth')
