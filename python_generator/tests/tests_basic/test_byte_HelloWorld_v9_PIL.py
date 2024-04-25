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
    qr_code_hw  = qr_gen_test.gen_QR_byte_encoding('HelloWorld', ec='Q')
    
    qr_code_hw_v9 = qr_code.QR_Code(                                         \
                    i_qr_ver        = 9,                                     \
                    c_ec            = 'Q',                                   \
                    m_data          = qr_code_hw.data_matrix,                \
                    i_num_codewords = qr_gen_test.v_to_num_codewords[9],     \
                    i_num_blocks    = qr_gen_test.v_to_blocks[9]['Q'],       \
                    i_ec_per_block  = qr_gen_test.v_to_ec_per_block[9]['Q'], \
                    l_alignment     = qr_gen_test.v_to_alignment[9]          \
    )
    
    # #-------#
    # #  PIL  #
    # #-------#
    
    # # Add whitespace
    # m_thick = np.zeros((qr_code_hw.side_length+8, qr_code_hw.side_length+8), dtype=bool)
    # m_thick[4:-4,4:-4] = qr_code_hw.cell_list_masked[qr_code_hw.best_qr]
    
    # # More pixels for clarity...
    # m_thick = np.repeat(m_thick, 5, axis=0)
    # m_thick = np.repeat(m_thick, 5, axis=1)
    
    # # Display
    # qr_image = img.fromarray(np.invert(m_thick))
    # qr_image.show()
    
    #----------#
    #  PIL v9  #
    #----------#
    
    # Add whitespace
    m_thick = np.zeros((qr_code_hw_v9.side_length+8, qr_code_hw_v9.side_length+8), dtype=bool)
    m_thick[4:-4,4:-4] = qr_code_hw_v9.cell_list_masked[qr_code_hw_v9.best_qr]
    
    # More pixels for clarity...
    m_thick = np.repeat(m_thick, 5, axis=0)
    m_thick = np.repeat(m_thick, 5, axis=1)
    
    # Display
    qr_image = img.fromarray(np.invert(m_thick))
    qr_image.show()