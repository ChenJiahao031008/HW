import math
import helper
import posenet
import numpy as np
from keras.optimizers import Adam
import cv2

if __name__ == "__main__":
    # Test model
    model = posenet.create_posenet()
    model.load_weights('./weights/checkpoint_weights.h5')
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipvalue=2.0)
    model.compile(optimizer=adam, loss={'cls1_fc_pose_xyz': posenet.euc_loss1x, 'cls1_fc_pose_wpqr': posenet.euc_loss1q,
                                        'cls2_fc_pose_xyz': posenet.euc_loss2x, 'cls2_fc_pose_wpqr': posenet.euc_loss2q,
                                        'cls3_fc_pose_xyz': posenet.euc_loss3x, 'cls3_fc_pose_wpqr': posenet.euc_loss3q})

    # model.compile(optimizer=adam, loss={'cls1': posenet.line_loss1, 
    #                                     'cls2': posenet.line_loss2, 'cls3': posenet.line_loss3})


    dataset_train, dataset_test, images_path = helper.getKingsTest()

    X_test = np.squeeze(np.array(dataset_test.images))
    y_test = np.squeeze(np.array(dataset_test.poses))

    testPredict = model.predict(X_test)

    valsx = testPredict[4]
    valsq = testPredict[5]

    # valsx = testPredict[2][:,0:1]
    # valsq = testPredict[2][:,0:1]

    # Get results... :/
    results = np.zeros((len(dataset_test.images),2))
    for i in range(len(dataset_test.images)):

        pose_x= np.asarray(dataset_test.poses[i][0:1])
        pose_q= np.asarray(dataset_test.poses[i][1:2])

        predicted_x = valsx[i]
        predicted_q = valsq[i]

        pose_x = np.squeeze(pose_x)
        pose_q = np.squeeze(pose_q)

        predicted_x = np.squeeze(predicted_x)*224
        predicted_q = np.squeeze(predicted_q)*224

        image = cv2.imread(images_path[i])
        print(images_path[i])

        cv2.line(image, (predicted_x[0],predicted_x[1]), (predicted_q[0],predicted_q[1]), (0, 0, 255), 3)

        real_rho = pose_x*112.0
        real_sin_theta = pose_q
        real_cos_theta = (1-real_sin_theta**2)**0.5
        real_py1 = -real_cos_theta*(1.0/real_sin_theta)*(-112) + real_rho*(1/real_sin_theta)
        real_py2 = -real_cos_theta*(1.0/real_sin_theta)*( 112) + real_rho*(1/real_sin_theta)
        real_py1 = int(112 - real_py1)
        real_py2 = int(112 - real_py2)
        cv2.line(image, (0,real_py1), (224,real_py2), (0, 255, 0), 3)
        name = "/home/demo/桌面/DL/Line-regression/keras-posenet/test/" + str(i) + ".png"
        cv2.imwrite(name, image)
        cv2.imshow("image",image)
        cv2.waitKey(0)
        print("real rho , sin_theta: ", real_rho, real_sin_theta)
        print("predict rho , sin_theta: ", predicted_x, predicted_q)

    #     #Compute Individual Sample Error
        
    #     results[i,:] = [error_x,theta]
    #     print 'Iteration:  ', i, '  Error XYZ (m):  ', error_x, '  Error Q (degrees):  ', theta
    # median_result = np.median(results,axis=0)
    # print('Median error ', median_result[0], 'm  and ', median_result[1], 'degrees.')