import fsps
from sedpy import observate
from read_images import unconver_read_filters
import time

def make_mock_images():
    _, sedpy_filts = unconver_read_filters()

    t0 = time.time()
    print('Generating stellar pop...')
    sp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1,
                                    sfh=0, logzsol=0.0, dust_type=2, dust2=0.2)
    t1 = time.time()
    print('Getting spectrum 1...')
    wave_aa, spec = sp.get_spectrum(tage=13, peraa=True)
    t2 = time.time()
    print('Getting spectrum 2...')
    wave_aa2, spec2 = sp.get_spectrum(zmet=2, tage=13, peraa=True)
    t3 = time.time()

    def print_time(name, t_start, t_end):
        print(f'Time for {name}: {t_end-t_start}')
    


    mags = observate.getSED(wave_aa, spec2, filterlist=sedpy_filts)
    fluxes_jy = 3631*10**(mags/-2.5)
    t4 = time.time()
    

    print_time('stellar pop', t0, t1) # 15 seconds
    print_time('spec 1', t1, t2) # 70 seconds
    print_time('spec 2', t2, t3) # fast
    print_time('mags', t3, t4) # fast


    # Next need to draw a circle, thens tart generating spectra in each pixel with slightly different stellar pops
    

if __name__ == '__main__':
    make_mock_images()