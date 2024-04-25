import sys
sys.path.append('../../src/')

import datetime as dt
import numpy as np
import PIL.Image as img
import PIL.ImageDraw as imgdraw

import qr_code_encode
import qr_code_gen

#==============================================================================#
# Main
#==============================================================================#
if __name__ == '__main__' :

    #-------------------------------------------------------#
    #  "QR Code Generator" instance --> generate "QR Code"  #
    #-------------------------------------------------------#
    qr_gen = qr_code_gen.QR_Code_Gen()
    qr_str = qr_code_encode.QR_Code_Encode()
        
    qr_code_contact = qr_gen.gen_QR_byte_encoding( qr_str.gen_str_contact(name="TestName", cell="+11234567890", anniversary=dt.datetime(2010,12,13), bday=dt.datetime(1985,1,13), email="test@test.test.gofuckyourself", note="Hello World") )
    print('\n'+qr_code_contact.get_string_info())
    
    #-------#
    #  PIL  #
    #-------#
    
    # Add whitespace
    m_thick = np.zeros((qr_code_contact.side_length+8, qr_code_contact.side_length+8), dtype=bool)
    m_thick[4:-4,4:-4] = qr_code_contact.cell_list_masked[qr_code_contact.best_qr]
    
    # More pixels for clarity...
    m_thick = np.repeat(m_thick, 5, axis=0)
    m_thick = np.repeat(m_thick, 5, axis=1)
    
    # Display
    qr_image = img.fromarray(np.invert(m_thick))
    qr_image.show()