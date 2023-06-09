text = open('./TeacherBNgetbest.txt', 'r')
uciqe = 0
uiqm = 0
ssim = 0
psnr = 0
flops = 0
all_ifno = list()
for i in range(5000):
    b = text.readline()
    if not b:
        break
    info = b.split('---')
    # if float(info[-6].split(':')[1][1:5]) < 50.5:
    #     flops += 1
    #     print('flops')
    if float(info[-1].split(':')[1][0:5]) > 30.5:
        uciqe += 1
        # print('uciqe')
        if float(info[-2].split(':')[1][1:5]) > 0.7:
            uiqm += 1
            # print('uiqm')
            if float(info[-3].split(':')[1][1:5]) > 0.66:
                ssim += 1
                # print('ssim')
                if float(info[-4].split(':')[1][1:5]) > 18:
                    psnr += 1
                    # print('psnr')
                    all_ifno.append(b)
                    # print(b)

print("flops:{}---uciqe:{}---uiqm:{}---ssim:{}---psnr:{}".format(flops, uciqe, uiqm, ssim, psnr))
for info_ in all_ifno:
    print(info_)
