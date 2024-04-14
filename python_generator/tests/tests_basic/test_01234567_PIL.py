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
    
    # #--------------------------------#
    # #  "QR Code" without __init__()  #
    # #--------------------------------#
    # l_data_poly = [16, 32, 12, 86, 97, 128, 236, 17, 236, 17, 236, 17, 236, 17, 236, 17]
    # l_expected  = [165, 36, 212, 193, 237, 54, 199, 135, 44, 85]
    # m_ec = QR_Code.get_rs_ec_remainder(None, np.array(l_data_poly, dtype=np.uint8), 10)
    # print(m_ec)

    #-------------------------------------------------------#
    #  "QR Code Generator" instance --> generate "QR Code"  #
    #-------------------------------------------------------#
    qr_gen_test      = qr_code_gen.QR_Code_Gen()
    qr_code_01234567 = qr_gen_test.gen_QR_numeric_encoding('01234567', ec='M')
    
    print(qr_code_01234567.get_string_info())
    print('\nScores:\t',qr_code_01234567.cell_list_score)
    
    #-------#
    #  PIL  #
    #-------#
    
    # Add whitespace
    m_thick = np.zeros((qr_code_01234567.side_length+8, qr_code_01234567.side_length+8), dtype=bool)
    m_thick[4:-4,4:-4] = qr_code_01234567.cell_list_masked[qr_code_01234567.best_qr]
    
    # More pixels for clarity...
    m_thick = np.repeat(m_thick, 5, axis=0)
    m_thick = np.repeat(m_thick, 5, axis=1)
    
    # Display
    qr_image = img.fromarray(np.invert(m_thick))
    qr_image.show()