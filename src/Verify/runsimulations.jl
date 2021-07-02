using Printf

configuration = [[ 30; 110.0; 9700.0; 14.0;11.0; -9.0], # 1
                 [ 40;  50.0; 2600.0;  5.0; 1.0; -7.0], # 2
                 [ 50;  80.0; 4900.0; 20.0; 2.0; -5.5], # 3
                 [ 60;  30.0; 8600.0; 30.0; 2.0; -3.0], # 4
                 [ 70; 150.0; 5000.0;  7.0; 2.0; -9.0], # 5
                 [ 80;  90.0; 3600.0;  4.0;20.0; -7.5], # 6
                 [160;  90.0; 3600.0;  4.0;20.0; -7.5], # 7
                 [120;  90.0; 3600.0;  4.0;20.0; -7.5], # 8
                 [180;  70.0; 9000.0; 10.0;10.0; -2.5], # 9
                 [100;  50.0; 7000.0; 12.0;12.0; -5.0], # 10
                 [200;  50.0; 7000.0; 12.0;12.0; -5.0], # 11
                 [150;  56.0; 2200.0; 12.0;12.0; -5.0], # 12
                 [100;  50.0; 4000.0;  3.0; 3.0; -7.0], # 13
                 [100;  50.0; 5000.0;  3.0; 3.0; -1.0], # 14
                 [100;  50.0; 3000.0;  2.0; 4.0; -2.0], # 15
                 [100;  50.0; 5500.0;  2.0; 5.0; -3.0], # 16
                 [300;  30.0; 6600.0; 30.0; 2.0; -3.0], # 17
                 [100;  30.0; 6700.0; 15.0; 2.0; -3.0], # 18
                 [100;  70.0; 6700.0; 11.0; 2.0; -3.0], # 19
                 [100;  70.0; 6700.0; 11.0; 2.0;-10.0], # 20
                 [100;  70.0; 6700.0; 16.0; 2.0;-11.0], # 21
                 [110; 110.0; 6000.0; 19.0; 2.0; -9.0], # 22
                 [100;  80.0; 7100.0;  5.0; 2.0; -8.0]] # 23


for i in 1:length(configuration)

    N  = round(Int, configuration[i][1])
    SN = configuration[i][2]
    T  = configuration[i][3]
    F  = configuration[i][4]
    S  = configuration[i][5]
    LS = configuration[i][6]

    @show configuration[i]

    t, flux, σ, idx, regflux, regσ, line, _ = simulatedata(N=N, SN=SN, Temperature=T, offset=F, scale=S, logs=LS, seed=i);

    figure(3)
    savefig(@sprintf("total_fluxes_%d.png", i))
    figure(4)
    savefig(@sprintf("flux_plot_%d.png", i))

    posterior = bpca(regflux', (regσ)',maxiter=600)

    g = unitgalaxyvector(T)*S

    randx = noisyintersectionvi(posterior=posterior, g=g)

    xavg = mean([randx() for i in 1:10_000])

    figure(20)
    cla()

    subplot(611)
    cla()
    plt.hist([randx()[1] for i=1:100_000], 30)
    plot(g[1], 0, "ro", label="true coordinate 1")
    plot(xavg[1], 0, "ko", label="xavg 1")
    legend()

    subplot(612)
    cla()
    plt.hist([randx()[2] for i=1:100_000], 30)
    plot(g[2], 0, "ro", label="true coordinate 2")
    plot(xavg[2], 0, "ko", label="xavg 2")
    legend()

    subplot(613)
    cla()
    plt.hist([randx()[3] for i=1:100_000], 30)
    plot(g[3], 0, "ro", label="true coordinate 3")
    plot(xavg[3], 0, "ko", label="xavg 3")
    legend()

    subplot(614)
    cla()
    plt.hist([randx()[4] for i=1:100_000], 30)
    plot(g[4], 0, "ro", label="true coordinate 4")
    legend()

    subplot(615)
    cla()
    plt.hist([randx()[5] for i=1:100_000], 30)
    plot(g[5], 0, "ro", label="true coordinate 5")
    legend()

    subplot(616)
    cla()
    plt.hist([randx()[6] for i=1:100_000], 30)
    plot(g[6], 0, "ro", label="true coordinate 6")
    legend()

    savefig(@sprintf("hist_%d.png", i))

end
