import sys
sys.path.append('../../src/')

import numpy as np
import PIL.Image as img
import PIL.ImageDraw as imgdraw

import qr_code_gen
import qr_code_encode

#==============================================================================#
# Main
#==============================================================================#
if __name__ == '__main__' :

    #-------------------------------------------------------#
    #  "QR Code Generator" instance --> generate "QR Code"  #
    #-------------------------------------------------------#
    qr_gen = qr_code_gen.QR_Code_Gen()
    qr_str = qr_code_encode.QR_Code_Encode()
    
    qr_code_wifi = qr_gen.gen_QR_byte_encoding( qr_str.gen_str_wifi("WPA", "WiFi Name", "password") )
    print('\n'+qr_code_wifi.get_string_info())

    #-------#
    #  PIL  #
    #-------#
    
    # Add whitespace
    m_thick = np.zeros((qr_code_wifi.side_length+8, qr_code_wifi.side_length+8), dtype=bool)
    m_thick[4:-4,4:-4] = qr_code_wifi.cell_list_masked[qr_code_wifi.best_qr]
    
    # More pixels for clarity...
    m_thick = np.repeat(m_thick, 5, axis=0)
    m_thick = np.repeat(m_thick, 5, axis=1)
    
    # Display
    qr_image = img.fromarray(np.invert(m_thick))
    qr_image.show()
    # qr_image.save("wifi_qr_0.png")