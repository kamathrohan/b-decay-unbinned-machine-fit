import b_meson_fit.mass as mass
import b_meson_fit as bmf
import b_meson_fit.background as background

signal_coeffs = bmf.coeffs.signal(bmf.coeffs.SM)
coeffs =[i.numpy() for i in signal_coeffs] 
events = bmf.signal.generate_signal_mass(signal_coeffs, events_total=10000).numpy()
print(events)