

####################################### to plot tidycache and notidycache results ######################################

    logdir = 'log/Completed logs/log_2021-04-21-161649_tidycache'

    fig = plt.figure(figsize=(15, 15), dpi=100)
    tidy = []
    files = os.listdir(logdir)
    for filename in files:
        print(filename)
        if (filename[:3] == 'dsr') and (filename[-7:-4] != 'hof'):
            data = pd.read_csv(logdir + '/' + filename)
            plt.plot(data['base_r_best'], color='C0')
            tidy.append(data['base_r_best'].values)

    logdir = 'log/Completed logs/log_2021-04-21-142035_notidycache'
    notidy = []
    files = os.listdir(logdir)
    for filename in files:
        print(filename)
        if (filename[:3] == 'dsr') and (filename[-7:-4] != 'hof'):
            data = pd.read_csv(logdir + '/' + filename)
            plt.plot(data['base_r_best'], color='C1')
            notidy.append(data['base_r_best'].values)





    ## comparison of different error metrics

    expression = 'x2*(-0.7289507583079632*x1 + 0.076926414459202349*x1/(x3 + x5) - 0.095542880884756653)'

    with open('config_kDeficit.json', encoding='utf-8') as f:
        config = json.load(f)

    X, y = load_frozen_RANS_dataset(config['task'])

    y_hat = eval_expression(expression, X)

    n_rep = 1

    print(f'{n_rep} evaluations of reward for different error metrics')

    starttime = time.time()
    for _ in range(n_rep):
        # Negative mean squared error
        # Range: [-inf, 0]
        # Value = -var(y) when y_hat == mean(y)
        ans = -np.mean((y - y_hat) ** 2)

    print(f'neg_mse took: {round(time.time() - starttime, 2)}')
    print(ans)

    starttime = time.time()
    for _ in range(n_rep):
        # Negative mean squared error
        # Range: [-inf, 0]
        # Value = -var(y) when y_hat == mean(y)
        ans = -np.sqrt(np.mean((y - y_hat) ** 2))
    print(f'neg_rmse took: {round(time.time() - starttime, 2)}')
    print(ans)

    starttime = time.time()
    for _ in range(n_rep):
        # Negative mean squared error
        # Range: [-inf, 0]
        # Value = -var(y) when y_hat == mean(y)
        var_y = np.std(y)
        ans = -np.mean((y - y_hat) ** 2) / var_y
    print(f'neg_nmse took: {round(time.time() - starttime, 2)}')
    print(ans)

    starttime = time.time()
    for _ in range(n_rep):
        # Negative mean squared error
        # Range: [-inf, 0]
        # Value = -var(y) when y_hat == mean(y)
        var_y = np.std(y)
        ans = -np.sqrt(np.mean((y - y_hat) ** 2) / var_y)
    print(f'neg_nrmse took: {round(time.time() - starttime, 2)}')

    starttime = time.time()
    for _ in range(n_rep):
        # Negative mean squared error
        # Range: [-inf, 0]
        # Value = -var(y) when y_hat == mean(y)
        ans = -np.log(1 + np.mean((y - y_hat) ** 2))
    print(f'neglog_mse took: {round(time.time() - starttime, 2)}')
    print(ans)

    args = [1.]
    starttime = time.time()
    for _ in range(n_rep):
        # Negative mean squared error
        # Range: [-inf, 0]
        # Value = -var(y) when y_hat == mean(y)
        ans = 1 / (1 + args[0] * np.mean((y - y_hat) ** 2))
    print(f'inv_mse took: {round(time.time() - starttime, 2)}')

    args = [1.]
    starttime = time.time()
    for _ in range(n_rep):
        # Negative mean squared error
        # Range: [-inf, 0]
        # Value = -var(y) when y_hat == mean(y)
        var_y = np.std(y)
        ans = 1 / (1 + args[0] * np.mean((y - y_hat) ** 2) / var_y)
    print(f'inv_nmse took: {round(time.time() - starttime, 2)}')
    print(ans)

    args = [1.]
    starttime = time.time()
    for _ in range(n_rep):
        # Negative mean squared error
        # Range: [-inf, 0]
        # Value = -var(y) when y_hat == mean(y)
        var_y = np.std(y)
        ans = 1 / (1 + args[0] * np.sqrt(np.mean((y - y_hat) ** 2) / var_y))
    print(f'inv_nrmse took: {round(time.time() - starttime, 2)}, with var_y IN the loop')
    print(ans)

    args = [1.]
    starttime = time.time()
    var_y = np.std(y)
    for _ in range(n_rep):
        # Negative mean squared error
        # Range: [-inf, 0]
        # Value = -var(y) when y_hat == mean(y)
        ans = 1 / (1 + args[0] * np.sqrt(np.mean((y - y_hat) ** 2) / var_y))
    print(f'inv_nrmse took: {round(time.time() - starttime, 2)}, with var_y OUT the loop')
    print(ans)

    starttime = time.time()
    for _ in range(n_rep):
        # Negative mean squared error
        # Range: [-inf, 0]
        # Value = -var(y) when y_hat == mean(y)
        ans = -np.mean((y - y_hat) ** 2 / np.sqrt(0.001 ** 2 + y ** 2))
    print(f'reg_mspe: {round(time.time() - starttime, 2)}')
    print(ans)

