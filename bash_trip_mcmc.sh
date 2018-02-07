make 

# (1) GR src file
# (2) 2^{p} (# bins) 
# (3) omegabar 
# (4) true src data file 
# (5) Target SNR 
# (6) seed
# (7) NBURN
# (8) NMCMC
# (9) chain file
# (10) maximum posterior src file
# (11) NCOOL

./mcmc high_src.dat 10 0. true_triple.dat 15. 1 100000 100000 chain.dat max_post_src.dat 30000

# valgrind --tool=memcheck --leak-check=yes --track-origins=yes ./mcmc high_src.dat 9 0. true_triple.dat 15. 1 200
# lldb ./mcmc high_src.dat 9 0. true_triple.dat 15. 1 100