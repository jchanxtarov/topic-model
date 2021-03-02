from model import PLSA

# TODO:
# if __name__ == '__main__':
#     parameter = (Z-1) + Z*(len(df["new_card_id"].unique())
#                            ) + Z*(len(df["new_shop_id"].unique()))
#     print('complete prep !')

#     time_start = time.time()
#     print('************plsa start*************')

#     plsa = PLSA(card, shop, Z)
#     LL = plsa.train()
#     AIC = (-2)*np.log(LL)+2*parameter
#     print('train finished, LL= ', LL)

#     plsa_all_time = time.time() - time_start
#     hour, minute, second = plsa_all_time / \
#         3600, (plsa_all_time % 3600)/60, plsa_all_time % 60
#     print("K = ", Z, "\nnow : ", int(hour), "-",
#           int(minute), "-", int(second), " (finished!)")

#     print('AIC= ', AIC)
#     time_now = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
#     np.savetxt('model_z' + str(Z) + '_pz_' +
#                time_now + ".csv", plsa.Pz, delimiter=',')
#     np.savetxt('model_z' + str(Z) + '_pcz_' + time_now +
#                ".csv", plsa.Pc_z, delimiter=',')
#     np.savetxt('model_z' + str(Z) + '_psz_' + time_now +
#                ".csv", plsa.Ps_z, delimiter=',')
#     print(np.sum(plsa.Pc_z))  # 確認
#     print(np.sum(plsa.Ps_z))  # 確認

#     result = np.array([LL, AIC])
#     np.savetxt('model_z' + str(K) + "_LL_AIC_" +
#                time_now + ".csv", result, delimiter=',')
