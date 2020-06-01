import numpy as np 
import time
import matplotlib.pyplot as plt
from main_function import build_geo, check_dir, fbp
from main_function import ssim_mse_psnr
from init import InitParser
from datasets import BuildDataSet
from idea_function import SinoInter


def test_f(sinogram_LineInter, geo_full, image_full):
    weg_num = 100
    wegs = np.linspace(1.5, 6, weg_num)
    ssim_max = 0
    mse_min = 10000
    psnr_max = 0

    ssim_wag = -1
    mse_wag = -1
    psnr_wag = -1

    for i,weg in enumerate(wegs):
        print("\n***********Now Testing {}/{}, Weg:{}***********".format(i+1, weg_num, weg))
        sinogram_new = SinoInter(sinogram_LineInter, geo_full, weg, option="Other")
        image_new = fbp(sinogram_new, geo_full)
        ssim, mse, psnr = ssim_mse_psnr(image_full, image_new)
        print("Ssim:{:.6f} Mse:{:.6f} Psnr:{:.6f}".format(ssim, mse, psnr))
        if ssim>ssim_max:
            print("Ssim Max-->weg={:.6f}, Ssim={:.6f}".format(weg, ssim))
            ssim_max = ssim
            ssim_wag = weg
        if mse<mse_min:
            print(" Mse Min-->weg={:.6f},  Mse={:.6f}".format(weg, mse))
            mse_min = mse
            mse_wag = weg
        if psnr>psnr_max:
            print("Psnr Max-->weg={:.6f}, Psnr={:.6f}".format(weg, psnr))
            psnr_max = psnr
            psnr_wag = weg 
    print("*"*10, "Result", "*"*10)
    print("Ssim_max:{:.6f} Weg:{:.6f}".format(ssim_max, ssim_wag))
    print("Mse_min:{:.6f} Weg:{:.6f}".format(mse_min, mse_wag))
    print("Psnr_max:{:.6f} Weg:{:.6f}".format(psnr_max, psnr_wag))
    return psnr_wag


def main(args):
    if args.use_cuda:
        print("Using GPU")
        # torch.cuda.set_device(args.gpu_id)
    else:
        print("Using CPU")

    check_dir(args.result_path)

    geo_full = build_geo(args.full_view)
    geo_sparse = build_geo(args.sparse_view)
    index = np.random.randint(0,223)


    datasets = {"train": BuildDataSet(args.data_root_path, args.train_folder, geo_full, geo_sparse, "train"),
                "val": BuildDataSet(args.data_root_path, args.val_folder, geo_full, geo_sparse, "val"),
                "test": BuildDataSet(args.data_root_path, args.test_folder, geo_full, geo_sparse, "test")}
    image, sinogram_full, sinogram_sparse, sinogram_LineInter, image_name = datasets["train"][index]
    res_sinogram = sinogram_full - sinogram_LineInter

    image_full = fbp(sinogram_full, geo_full)
    image_sparse = fbp(sinogram_sparse, geo_sparse)
    plt.figure()
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image_sparse, "gray")
    # plt.title("sinogram_sparse")
    plt.show()
    
    # image_LineInter = fbp(sinogram_LineInter, geo_full)

    # ssim_0, mse_0, psnr_0 = ssim_mse_psnr(image, image_sparse)
    # ssim_1, mse_1, psnr_1 = ssim_mse_psnr(image, image_LineInter)
    # print("   Sparse Image--> SSIM:{}, MSE:{}, PSNR:{}".format(ssim_0, mse_0, psnr_0))
    # print("LineInter Image--> SSIM:{}, MSE:{}, PSNR:{}".format(ssim_1, mse_1, psnr_1))
    # time.sleep(5)

    # weg = test_f(sinogram_LineInter, geo_full, image)

    # sinogram_new = SinoInter(sinogram_LineInter, geo_full, weg) 
    # sinogram_res = sinogram_full - sinogram_new
    # image_new = fbp(sinogram_new, geo_full)

    # image_res = image_full - image_new

    # plt.figure()
    # plt.suptitle(image_name)
    # plt.subplot(231),plt.xticks([]),plt.yticks([]),plt.imshow(sinogram_full, "gray"),       plt.title("sinogram_full")
    # plt.subplot(232),plt.xticks([]),plt.yticks([]),plt.imshow(sinogram_sparse, "gray"),     plt.title("sinogram_sparse")
    # plt.subplot(233),plt.xticks([]),plt.yticks([]),plt.imshow(sinogram_LineInter, "gray"),  plt.title("sinogram_LineInter")
    # # plt.subplot(234),plt.xticks([]),plt.yticks([]),plt.imshow(res_sinogram, "gray"),        plt.title("res_sinogram")
    # # plt.subplot(235),plt.xticks([]),plt.yticks([]),plt.imshow(sinogram_new, "gray"),        plt.title("sinogram_new")
    # # plt.subplot(236),plt.xticks([]),plt.yticks([]),plt.imshow(sinogram_res, "gray"),        plt.title("sinogram_res")
    # plt.show()
    # plt.figure()
    # plt.suptitle(image_name)
    # plt.subplot(231),plt.xticks([]),plt.yticks([]),plt.imshow(image, "gray"),           plt.title("image")
    # plt.subplot(232),plt.xticks([]),plt.yticks([]),plt.imshow(image_full, "gray"),      plt.title("image_full")
    # plt.subplot(233),plt.xticks([]),plt.yticks([]),plt.imshow(image_sparse, "gray"),    plt.title("image_sparse")
    # plt.subplot(234),plt.xticks([]),plt.yticks([]),plt.imshow(image_LineInter, "gray"), plt.title("image_LineInter")
    # plt.subplot(235),plt.xticks([]),plt.yticks([]),plt.imshow(image_new, "gray"),       plt.title("image_new")
    # plt.subplot(236),plt.xticks([]),plt.yticks([]),plt.imshow(image_res, "gray"),    plt.title("image_res")
    # plt.show()

    # ssim_0, mse_0, psnr_0 = ssim_mse_psnr(image_full, image_sparse)
    # ssim_1, mse_1, psnr_1 = ssim_mse_psnr(image_full, image_LineInter)
    # ssim_2, mse_2, psnr_2 = ssim_mse_psnr(image_full, image_new)
    # ssim_0, mse_0, psnr_0 = ssim_mse_psnr(image, image_sparse)
    # ssim_1, mse_1, psnr_1 = ssim_mse_psnr(image, image_LineInter)
    # ssim_2, mse_2, psnr_2 = ssim_mse_psnr(image, image_new)
    # print("   Sparse Image--> SSIM:{}, MSE:{}, PSNR:{}".format(ssim_0, mse_0, psnr_0))
    # print("LineInter Image--> SSIM:{}, MSE:{}, PSNR:{}".format(ssim_1, mse_1, psnr_1))
    # print("      New Image--> SSIM:{}, MSE:{}, PSNR:{}".format(ssim_2, mse_2, psnr_2))
    print("Run Done")

if __name__ == "__main__":
    parsers = InitParser()
    main(parsers)