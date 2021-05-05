

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