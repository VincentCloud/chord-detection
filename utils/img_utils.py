import matplotlib.pyplot as plt
import numpy as np
import skimage.draw as draw
import torch
import torchvision.transforms as transforms
from PIL import Image
from scipy import ndimage
from scipy.ndimage.measurements import center_of_mass, label
from skimage.feature import peak_local_max
from pathlib import Path
import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def rescale(image, size):
    h, w = image.shape[-2:]
    if isinstance(size, int):
        if h < w:
            new_h, new_w = size * h / w, size
        else:
            new_h, new_w = size, size * w / h
    else:
        new_h, new_w = size

    new_h, new_w = int(new_h), int(new_w)
    resize = transforms.Resize((new_h, new_w))
    img = transforms.ToTensor()(resize(transforms.ToPILImage()(image)))

    return img


def compute_gradient(image):
    # we compute the gradient of the image
    '''kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        sx = ndimage.convolve(depth[0][0], kx)
        sy = ndimage.convolve(depth[0][0], ky)'''
    sx = ndimage.sobel(image, axis=0, mode='nearest')
    sy = ndimage.sobel(image, axis=1, mode='nearest')
    gradient = transforms.ToTensor()(np.hypot(sx, sy))

    return gradient[0]


def local_max(image, min_dist, t_rel):
    max_out = peak_local_max(image, min_distance=min_dist, threshold_rel=t_rel, exclude_border=False, indices=False)
    labels_out = label(max_out)[0]
    max_out = np.array(center_of_mass(max_out, labels_out, range(1, np.max(labels_out) + 1))).astype(np.int)
    max_values = []

    for index in max_out:
        max_values.append(image[index[0]][index[1]])

    max_out = np.array([x for _, x in sorted(zip(max_values, max_out), reverse=True, key=lambda x: x[0])])

    return max_out


def corner_mask(output, min_dist):
    max_coord = local_max(output, min_dist, t_rel=0.3)
    corners = torch.zeros(3, output.shape[0], output.shape[1])
    for idx, (i, j) in enumerate(max_coord):
        cx, cy = draw.circle_perimeter(i, j, 9, shape=output.shape)
        if idx < 4:
            corners[0, cx, cy] = 1.

    return corners, max_coord


def save_img(input, output, min_dist, name):
    corners, max_coord = corner_mask(output, min_dist)
    rgb, corners = transforms.ToPILImage()(input), transforms.ToPILImage()(corners)
    image = Image.blend(rgb, corners, 0.3)
    plt.ioff()
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches((15, 6))
    ax[1].axis('off')
    ax[1].set_title('RGB image')
    ax[0].axis('off')
    ax[0].set_title('Network\'s output')
    ax[0].imshow(output, cmap='afmhot', vmin=0, vmax=1)
    ax[1].imshow(rgb)
    for (x, y) in max_coord:
        ax[1].scatter(y, x)
    Path('output/').mkdir(parents=True, exist_ok=True)
    plt.savefig('output/{image}.png'.format(image=name))
    plt.close('all')
    print('Image {name} saved'.format(name=name))


def show_corners(image, corners):
    """Show image with landmarks"""
    plt.imshow(image, cmap='gray')
    plt.scatter(corners[:, 1], corners[:, 0], s=10, marker='.', c='r')
    plt.pause(0.005)  # pause a bit so that plots are updated


def single_frame_operation(img, mirror=False, output_size=300):
    top_orientation = get_string_orientation(img)
    top_orientation_in_degrees = top_orientation * (180.0 / 3.141592653589793238463)
    rotated_image = rotate_image(img, top_orientation_in_degrees)

    # image_to_be_annotated = rotated_image if mirror else cv2.flip(rotated_image, 1)
    result = get_hand_rectangle(rotated_image, mirror=mirror)
    annotated_with_hands, rectangle = result[0], result[1]
    if rectangle:
        rec_p1 = (rectangle['top left']['x'], rectangle['top left']['y'])
        rec_p2 = (rectangle['bottom right']['x'], rectangle['bottom right']['y'])
        cv2.rectangle(annotated_with_hands, rec_p1, rec_p2, (0, 244, 0))
        hand_rectangle = annotated_with_hands[rec_p1[1]: rec_p2[1], rec_p1[0]: rec_p2[0]]
        # hand_rectangle = pad_to_square(hand_rectangle)
        # hand_rectangle = cv2.resize(hand_rectangle, (output_size, output_size), interpolation=cv2.INTER_AREA)
        print(hand_rectangle.shape)
    else:
        hand_rectangle = np.zeros(img.shape)

    return hand_rectangle


def get_string_orientation(frame):
    edges = cv2.Canny(frame, 100, 150, None, 3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)
    # cdst = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

    # for each line, get their orientation
    orientation = np.arctan(np.float64(lines[:, :, 3] - lines[:, :, 1]) / np.float64(lines[:, :, 2] - lines[:, :, 0]))

    orientation = orientation.reshape(orientation.shape[0], )

    values, bins, patches = plt.hist(orientation, density=True, bins=60)

    # get the top two orientations
    indices = np.argsort(values)[::-1]
    top_orientation = (bins[indices[0]], bins[indices[0] + 1])

    # get the lines with orientations in such intervals
    interval1 = within_bin(orientation, top_orientation)
    line_filter = list(interval1)
    top_lines = lines[line_filter]
    top_orientations = orientation[line_filter]

    # hist, bins = np.histogram(orientation, bins=20)
    # if top_lines is not None:
    #     for i in range(0, len(top_lines)):
    #         l = top_lines[i][0]
    # cv.line(cdst, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

    # get the average value of the top orientation
    return np.mean(top_orientations)
    # plt.show()
    # return cv.flip(frame, 1)


def within_bin(orientation, or_bin):
    upper_bound = orientation < or_bin[1]
    lower_bound = orientation > or_bin[0]
    return upper_bound * lower_bound


def get_hand_rectangle(image, mirror=False):
    with mp_hands.Hands(
      static_image_mode=True,
      max_num_hands=1,
      min_detection_confidence=0.4) as hands:
        # Convert the BGR image to RGB before processing.
        if not mirror:
            image = cv2.flip(image, 1)
        # google mediapipe wants the frame mirrored
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_height, image_width, _ = image.shape
        # annotated_image = image.copy()
        if not results.multi_hand_landmarks:
            return (image, None) if mirror else (cv2.flip(image, 1), None)
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # print('hand_landmarks:', hand_landmarks)
            hand_handedness_dict = MessageToDict(results.multi_handedness[i])
            hand_handedness = hand_handedness_dict['classification'][0]['label']
            hand_rectangle = None
            if hand_handedness == 'Left':
                print('got a left hand')
                dict_obj = MessageToDict(hand_landmarks)
                h_l = dict_obj['landmark']
                hand_rectangle = crop_hand(image, h_l)
                # mp_drawing.draw_landmarks(
                #     image,
                #     hand_landmarks,
                #     mp_hands.HAND_CONNECTIONS,
                #     mp_drawing_styles.get_default_hand_landmarks_style(),
                #     mp_drawing_styles.get_default_hand_connections_style())

        # cv.imshow('image', cv2.flip(image, 1))
        # cv2.waitKey(0)
        return (image, hand_rectangle) if mirror else (cv2.flip(image, 1), hand_rectangle)


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def crop_hand(image_data, hand, margin=0.2):
    # cv.imshow('test_image', image_data)
    # cv.waitKey(0)
    img_width, img_height = image_data.shape[1], image_data.shape[0]
    x_rec_right = round(max(hand, key=lambda pos: pos['x'])['x'] * img_width)
    y_rec_up = round(min(hand, key=lambda pos: pos['y'])['y'] * img_height)
    y_rec_down = round(max(hand, key=lambda pos: pos['y'])['y'] * img_height)

    box_width = x_rec_right
    box_height = y_rec_down - y_rec_up

    # return mirrored output
    return {
        'bottom right': {
            'x': img_width - 1 - 0,
            'y': max(y_rec_down + round(box_height * margin * 3), box_height - 1)
        },
        'top left': {
            'x': img_width - 1 - x_rec_right - round(img_width * margin),
            'y': max(y_rec_up - round(box_height * margin * 3), 0)
        }
    }


def pad_to_square(cropped_img):
    # it is assumed that the width is
    h, w = cropped_img.shape[0], cropped_img.shape[1]
    color = [0, 0, 0]
    if h > w:
        # if height > width, pad width
        delta_w = h - w
        left, right = delta_w // 2, delta_w - delta_w // 2
        new_im = cv2.copyMakeBorder(cropped_img, 0, 0, left, right, cv2.BORDER_CONSTANT, value=color)
    else:
        delta_h = w - h
        top, bottom = delta_h // 2, delta_h - delta_h // 2
        new_im = cv2.copyMakeBorder(cropped_img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=color)
    return new_im
