import os
from glob import glob

import cv2
import math
import numpy as np



def DarkChannel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark


def AtmLight(im, dark):
    [h, w] = im.shape[:2]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = dark.reshape(imsz, 1);
    imvec = im.reshape(imsz, 3);

    indices = darkvec.argsort();
    indices = indices[imsz - numpx::]

    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A


def TransmissionEstimate(im, A, sz):
    omega = 0.95;
    im3 = np.empty(im.shape, im.dtype);

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]

    transmission = 1 - omega * DarkChannel(im3, sz);
    return transmission


def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r));
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r));
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r));
    cov_Ip = mean_Ip - mean_I * mean_p;

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r));
    var_I = mean_II - mean_I * mean_I;

    a = cov_Ip / (var_I + eps);
    b = mean_p - a * mean_I;

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r));
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r));

    q = mean_a * im + mean_b;
    return q;


def TransmissionRefine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY);
    gray = np.float64(gray) / 255;
    r = 60;
    eps = 0.0001;
    t = Guidedfilter(gray, et, r, eps);

    return t;


def Recover(im, t, A, tx=0.1):
    res = np.empty(im.shape, im.dtype);
    t = cv2.max(t, tx);

    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]

    return res


def process_image(input_path, output_folder):
    # 创建输出目录
    os.makedirs(output_folder, exist_ok=True)

    src = cv2.imread(input_path)
    if src is None:
        print(f"无法读取图像: {input_path}")
        return

    I = src.astype('float64') / 255
    dark = DarkChannel(I, 15)
    A = AtmLight(I, dark)
    te = TransmissionEstimate(I, A, 15)
    t = TransmissionRefine(src, te)
    J = Recover(I, t, A, 0.1)

    # 生成输出文件名
    base_name = os.path.basename(input_path)
    name, ext = os.path.splitext(base_name)

    # 保存去雾结果
    dehaze_path = os.path.join(output_folder, f"{name}_dehaze{ext}")
    dehaze_img = (np.clip(J, 0, 1) * 255).astype(np.uint8)  # 关键修改
    cv2.imwrite(dehaze_path, dehaze_img)
    # cv2.imwrite(dehaze_path, J * 255)



    # 保存对比图
    arr = np.hstack((I, J))
    contrast_path = os.path.join(output_folder, f"{name}_contrast{ext}")
    contrast_img = (np.clip(arr, 0, 1) * 255).astype(np.uint8)  # 关键修改
    cv2.imwrite(contrast_path, contrast_img)
    # cv2.imwrite(contrast_path, arr * 255)

    print(f"已处理: {input_path}")


if __name__ == '__main__':
    # 配置输入输出路径
    input_folder = "datasets/input_images"  # 输入文件夹路径
    output_folder = "datasets/output_images"  # 输出文件夹路径

    # 支持处理的图片格式
    extensions = ['jpg', 'jpeg', 'png', 'bmp']

    # 获取所有图片文件
    image_files = []
    for ext in extensions:
        image_files.extend(glob(os.path.join(input_folder, f'*.{ext}')))
        image_files.extend(glob(os.path.join(input_folder, f'*.{ext.upper()}')))

    # 处理所有图片
    for image_file in image_files:
        process_image(image_file, output_folder)

    print("批量处理完成！")

#
# if __name__ == '__main__':
#     import sys
#
#     try:
#         fn = sys.argv[1]
#     except:
#         fn = 'demo.jpg'
#
#
#     def nothing(*argv):
#         pass
#
#
#     src = cv2.imread(fn);
#     I = src.astype('float64') / 255;
#     dark = DarkChannel(I, 15);
#     A = AtmLight(I, dark);
#     te = TransmissionEstimate(I, A, 15);
#     t = TransmissionRefine(src, te);
#     J = Recover(I, t, A, 0.1);
#     arr = np.hstack((I, J))
#     # cv2.imshow("contrast", arr) 弹个界面
#     cv2.imwrite("dehaze.png", J * 255)
#     cv2.imwrite("contrast.png", arr * 255);
#     cv2.waitKey();
