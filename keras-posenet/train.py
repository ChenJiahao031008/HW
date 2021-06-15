import helper
import posenet
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

if __name__ == "__main__":
    # Variables
    batch_size = 16

    # Train model
    model = posenet.create_posenet( True) # GoogLeNet (Trained on Places)
    adam = Adam(lr=0.001, clipvalue=1.5)
    model.compile(optimizer=adam, loss={'cls1_fc_pose_xyz': posenet.euc_loss1x, 'cls1_fc_pose_wpqr': posenet.euc_loss1q,
                                        'cls2_fc_pose_xyz': posenet.euc_loss2x, 'cls2_fc_pose_wpqr': posenet.euc_loss2q,
                                        'cls3_fc_pose_xyz': posenet.euc_loss3x, 'cls3_fc_pose_wpqr': posenet.euc_loss3q})
    # model.compile(optimizer=adam, loss={'cls1': posenet.line_loss1,
    #                                     'cls2': posenet.line_loss2, 'cls3': posenet.line_loss3})

    dataset_train, dataset_test = helper.getKings()

    X_train = np.squeeze(np.array(dataset_train.images))
    y_train = np.squeeze(np.array(dataset_train.poses))

    y_train_x = y_train[:,0:1]
    y_train_q = y_train[:,1:2]

    X_test = np.squeeze(np.array(dataset_test.images))
    y_test = np.squeeze(np.array(dataset_test.poses))

    # two param
    y_test_x = y_test[:,0:1]
    y_test_q = y_test[:,1:2]


    ## four param
    # rho = y_test[:,0:1] * 112
    # sin_test = y_test[:,1:2]
    # cos_test = np.sqrt(1 - sin_test * sin_test)
    # p1_x = -112;
    # p2_x = 112;
    # p1_ls = []
    # p2_ls = []
    # for i in range(len(sin_test)):
    #     if (np.abs(sin_test[i])<1e-1):
    #         p1_y = p2_y = (112 - rho[i])/224;
    #         p1_x = 0.0;
    #         p2_x = 224/224.0;
    #     else:
    #         p1_y = cos_test[i]*(1.0/sin_test[i])*p1_x + rho[i]*(1/sin_test[i]);
    #         p2_y = cos_test[i]*(1.0/sin_test[i])*p2_x + rho[i]*(1/sin_test[i]);
    #         p1_x = 0;
    #         p2_x = 224.0;
    #         p1_y = (112 - p1_y)/224.0;
    #         p2_y = (112 - p2_y)/224.0;
    #     p1_ls.append([p1_x, p1_y])
    #     p2_ls.append([p2_x, p2_y])
    # y_test_x = np.array(p1_ls)
    # y_test_q = np.array(p2_ls)


    # Setup checkpointing
    checkpointer = ModelCheckpoint(filepath="checkpoint_weights.h5", verbose=1, save_best_only=True, save_weights_only=True)

    model.fit(X_train, [y_train_x, y_train_q, y_train_x, y_train_q, y_train_x, y_train_q],
          batch_size=batch_size,
          nb_epoch=20,
          validation_data=(X_test, [y_test_x, y_test_q, y_test_x, y_test_q, y_test_x, y_test_q]),
          callbacks=[checkpointer])


    # model.fit(X_train, [y_train, y_train, y_train],
    #       batch_size=batch_size,
    #       nb_epoch=20,
    #       validation_data=(X_test, [y_test, y_test, y_test]),
    #       callbacks=[checkpointer])

    model.save_weights("custom_trained_weights.h5")
