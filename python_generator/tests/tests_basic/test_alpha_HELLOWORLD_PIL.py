import sys
sys.path.append('../../src/')

import numpy as np
import PIL.Image as img
import PIL.ImageDraw as imgdraw

import qr_code
import qr_code_gen

#==============================================================================#
# Main
#==============================================================================#
if __name__ == '__main__' :

    #-------------------------------------------------------#
    #  "QR Code Generator" instance --> generate "QR Code"  #
    #-------------------------------------------------------#
    qr_gen_test = qr_code_gen.QR_Code_Gen()
    qr_code_hw  = qr_gen_test.gen_QR_alphanumeric_encoding('HELLO WORLD', ec='Q')
    
    print(qr_code_hw.get_string_info())
    
    #-------#
    #  PIL  #
    #-------#
    
    # Add whitespace
    m_thick = np.zeros((qr_code_hw.side_length+8, qr_code_hw.side_length+8), dtype=bool)
    m_thick[4:-4,4:-4] = qr_code_hw.cell_list_masked[qr_code_hw.best_qr]
    
    # More pixels for clarity...
    m_thick = np.repeat(m_thick, 5, axis=0)
    m_thick = np.repeat(m_thick, 5, axis=1)
    
    # Display
    qr_image = img.fromarray(np.invert(m_thick))
    qr_image.show()