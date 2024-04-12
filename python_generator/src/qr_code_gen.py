import numpy as np
import PIL.Image as img
import PIL.ImageDraw as imgdraw

#==============================================================================#
# QR Code Steps:
#   (1 & 2) Data analysis & encoding:
#       - Find format (whether basic alphanumeric or ECI) which supports input, and convert.
#       - Is possible to use multiple, if certain parts can be more efficiently coded then others.
#   (3) Error Correction Coding
#       - Divide data into blocks; generate EC codewords and append.
#   (4 & 5) Structure data and place. [Finder pattern, seperators, timing pattern, alignments.]
#   (6) Data Masking
#       - In order to balance the QR code, choose 1 mask option to optimize dark/light balance.
#   (7) [Format/version info.]
#==============================================================================#

#==============================================================================#
# Decoding is the reverse:
# (1) Get QR code image, read format info, assure correct orientation (rotated, mirrored).
# (2) Determine version of symbol, read version number if applicable.
# (3) Undo data masking of encoding region and rebuild data + error correction codewords.
# (4) Assure data with error correction, divide data into sigements according to mode indicators and character count indicators.
# (5) Decode data characters with mode in use.
#==============================================================================#

#------------------------------------------------------------------------------#
# Abbreves:
#   - BCH = Bose-Chaudhuri-Hocquenghem
#   - ECI = Extented Channel Interpretation
#   - RS  = Reed-Solomon [Error Correction] [L/M/Q/H]
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# Encodable Character Set
#   - Numerical Data (Just a big number!)
#       - Actually no, each number is stupidly broken into sections of 3 decimal digits.
#           - Then each grouping is individually converted to binary.
#   - Alphanumeric: 0-9 A-Z $ % * + - . / : (Space)
#   - Byte Data (obvi 8bit per)
#   - Kanji (13bit)
#------------------------------------------------------------------------------#
#   - Mixing (Within one QR Code.)
#   - Structured Append (Between multiple QR Codes.)
#------------------------------------------------------------------------------#
# FNC1 Mode:
#   - Messages containing specific data formats.
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# Default Interpretation for QR Code == ECI 000003 == ISO/IEC 8859-1 Character Set
#   - Use ECI protocol for other character sets
#   - Extended Channel Interpretations
#------------------------------------------------------------------------------#
















#==============================================================================#
# Data Scoring for regular QR code:
#   - Penalty points are as follows: (Weighted penalties: N1 = 3, N2 = 3, N3 = 40, N4 = 10)
#       (1) Adjacent modules in both row/col, 5 or more, in same color.
#           - Points: N1 + (# - 5)
#       (2) Each 2x2 block of same color. (3x3 block has 4 2x2 blocks!)
#           - Points: N2
#       (3) The finder pattern, any oscillating 1:1:3:1:1 pattern.
#           - Points: N3
#       (4) Proportion of light to dark. (Each 5% in a either direction adds N4 penalty.)
#           - 45% - 55% =  0 points.
#           - 40% - 60% = 10 points.
#           - 35% - 65% = 20 points.
#------------------------------------------------------------------------------#
# Data Scoring for Micro QR
#  - Rather than penalties, it is better to have more dark modules on the edge!
#      - SUM dark modules on bottom edge.
#      - SUM dark modules on right edge.
#      - Score = (SUM small)x16 + (SUM large)
#==============================================================================#

#==============================================================================#
# Beautification:
#   - A QR code needs 4 modules wide of empty space on each side.
#   - Micro QR code needs 2 modules wide.
#==============================================================================#


#------------------------------------------------------------------------------#

class QR_Code :

    def __init__(self, i_qr_ver, c_ec, m_data, i_num_codewords, i_num_blocks, i_ec_per_block, l_alignment):
        
        # QR Code info.
        self.version        = i_qr_ver # Version implies size of QR code.
        self.side_length    = 17+(i_qr_ver*4)
        self.err_corr_sel   = c_ec
        self.data_mask_sel  = None
        self.data_matrix    = m_data
        self.codewords      = i_num_codewords
        self.blocks         = i_num_blocks
        self.ec_per_block   = i_ec_per_block
        self.alignment_list = l_alignment
        
        # https://numpy.org/doc/stable/reference/routines.matlib.html
        self.cell_finished = np.zeros((self.side_length, self.side_length), dtype=bool)
        self.cell_shaded   = np.zeros((self.side_length, self.side_length), dtype=bool)
        
        # Function Patterns
        self.fill_finder_patterns()    #
        self.fill_timing_patterns()    # Oscillating strips that run between finder patterns.
        self.fill_alignment_patterns() # Places: [5x5 dark, 3x3 light, 1 dark center]
        
        # Info
        self.fill_version_info(i_qr_ver)
        self.fill_format_into(0) # Pre-emptive claim of format into...
        
        # Data + Final Mask
        self.m_final_data = None
        self.m_final_ec   = None
        self.m_final_qr   = None
        self.fill_data(m_data)
        
        self.cell_list_masked = [None]*8
        for ii in [2]:
            self.cell_list_masked[ii] = self.select_mask_fill_format(ii) # Will overwrite the '0' format info.
        
        self.test_print_info()
        #self.test_format_visual()
        self.test_display(self.cell_list_masked[2])
        
    #==================#
    #  Test Functions  #
    #==================#
    
    def test_format_visual(self):
        
        # Add whitespace
        m_thick = np.zeros((self.side_length+8, self.side_length+8), dtype=bool)
        m_thick[4:-4,4:-4] = self.cell_finished
        
        # More pixels for clarity...
        m_thick = np.repeat(m_thick, 5, axis=0)
        m_thick = np.repeat(m_thick, 5, axis=1)
        
        # Display
        qr_image = img.fromarray(np.invert(m_thick))
        qr_image.show()
    
    def test_display(self, m_shaded):
        
        # Add whitespace
        m_thick = np.zeros((self.side_length+8, self.side_length+8), dtype=bool)
        m_thick[4:-4,4:-4] = m_shaded
        
        # More pixels for clarity...
        m_thick = np.repeat(m_thick, 5, axis=0)
        m_thick = np.repeat(m_thick, 5, axis=1)
        
        # Display
        qr_image = img.fromarray(np.invert(m_thick))
        qr_image.show()
        
    def test_print_info(self):
    
        t_str = ""
        
        t_str += "\nVersion #                : "+str(self.version)
        t_str += "\nReed-Solomon EC Level    : "+str(self.err_corr_sel)
        t_str += "\n# Codewords in QR        : "+str(self.codewords)
        t_str += "\n# Blocks in QR           : "+str(self.blocks)
        t_str += "\n# EC Codewords per Block : "+str(self.ec_per_block)
        t_str += "\n# Alignment List         : "+str(self.alignment_list)
        
        t_str += "\n\nData Matrix:"+str(self.m_final_data)
        t_str += "\n\nEC Matrix:"+str(self.m_final_ec)
        #t_str += "\n\nFinal Matrix:"+str(self.m_final_qr)
        
        t_str += "\n"
    
        print(t_str)
        
        for ii in range(self.m_final_qr.size) :
            print(self.m_final_qr[ii],"\t:=\t",hex(self.m_final_qr[ii])," :=\t")
        
    #=====================#
    #  Support Functions  #
    #=====================#
    
    def qr_draw_pixel(self, value, start):
        row, col = start
        self.cell_finished[row][col] = True
        self.cell_shaded[row][col]   = value
    
    def qr_draw_row(self, value, start, size):
        row, col = start
        for ii in range(size):
            self.cell_finished[row][col+ii] = True
            self.cell_shaded[row][col+ii]   = value
    
    def qr_draw_col(self, value, start, size):
        row, col = start
        for ii in range(size):
            self.cell_finished[row+ii][col] = True
            self.cell_shaded[row+ii][col]   = value
            
    def qr_draw_square(self, value, start, size):
        row, col = start
        for ii in range(size-1):
        
            # Left Column (Down)
            self.cell_finished[row][col+ii] = True
            self.cell_shaded[row][col+ii]   = value
            
            # Bot Row (Left --> Right)
            self.cell_finished[row+ii][col+size-1] = True
            self.cell_shaded[row+ii][col+size-1]   = value
            
            # Right Column (Up)
            self.cell_finished[row+size-1][col+size-1-ii] = True
            self.cell_shaded[row+size-1][col+size-1-ii]   = value
            
            # Top Row (Right --> Left)
            self.cell_finished[row+size-1-ii][col] = True
            self.cell_shaded[row+size-1-ii][col]   = value
        
    def qr_draw_block(self, value, start, size):
        row, col = start
        for ii in range(size):
            for jj in range(size):
                self.cell_finished[row+ii][col+jj] = True
                self.cell_shaded[row+ii][col+jj]   = value
                
    def qr_draw_finder(self, start):
        row, col = start
        self.qr_draw_square(True, (row, col), 7)
        self.qr_draw_square(False, (row+1, col+1), 5)
        self.qr_draw_block(True, (row+2, col+2), 3)
        
    def qr_draw_alignment(self, start):
        row, col = start
        self.qr_draw_square(True, (row, col), 5)
        self.qr_draw_square(False, (row+1, col+1), 3)
        self.qr_draw_pixel(True, (row+2, col+2))
    
    #=======================#
    #  QR Functions - Data  #
    #=======================#
        
    def get_rs_ec_remainder(self, m_data, i_degree):
        
        #-----------------------------------------------------------------------
        # α^0 --> (2^0)
        # α^1 --> (2^1)
        # α^2 --> (2^2)
        # α^3 --> (2^3)
        # α^4 --> (2^4)
        # α^5 --> (2^5)
        # α^6 --> (2^6)
        # α^7 --> (2^7)
        #-----------------------------------------------------------------------
        # α^8  --> (2^4) + (2^3) + (2^2) + (2^0)  # prime modulus polynomial (x^8) + (x^4) + (x^3) + (x^2) + (x^0)
        # α^9  --> (2^5) + (2^4) + (2^3) + (2^1)
        # α^10 --> (2^6) + (2^5) + (2^4) + (2^2)
        # α^11 --> (2^7) + (2^6) + (2^5) + (2^3)
        #-----------------------------------------------------------------------
        # α^12 --> (2^7) + (2^6) + (2^4) + (2^4) + (2^3) + (2^2) + (2^0) # prime modulus polynomial (x^8) + (x^4) + (x^3) + (x^2) + (x^0)
        #          (2^7) + (2^6) + (2^3) + (2^2) + (2^0)
        #          [205]
        #-----------------------------------------------------------------------
        # α^25 --> (2^1) + (2^0)
        #-----------------------------------------------------------------------
        
        alpha_log_table = [None]*256
        alpha_antilog_table = [None]*256
        
        i_alpha = 1
        for ii in range(256):
            
            alpha_log_table[ii] = i_alpha
            alpha_antilog_table[i_alpha] = ii
            
            if(i_alpha > 127):
                i_alpha = (i_alpha << 1) ^ 0x11D
            else:
                i_alpha = (i_alpha << 1)
                
        #-----------------------------------------------------------------------
        # Each generator polynomial for "'n' EC codwords" is the product of the first degree polynomials (x - α^0)(x - α^1)...(x - α^(n-1))            
        #-----------------------------------------------------------------------
        
        ll_alpha_poly = [None]*(i_degree+1)
        ll_alpha_poly[1] = [0, 0]
        ll_alpha_poly[2] = [0, 25, 1]
        
        for ii in range(3,i_degree+1):
            ll_alpha_poly[ii] = [None]*(ii+1)
            
            ll_alpha_poly[ii][0] = 0
            
            for jj in range(1,ii):
                
                antilog0 = alpha_log_table[ ll_alpha_poly[ii-1][jj] ]
                antilog1 = alpha_log_table[ (ll_alpha_poly[ii-1][jj-1] + (ii-1)) % 255 ]
                
                ll_alpha_poly[ii][jj] = alpha_antilog_table[antilog0 ^ antilog1]
            
            ll_alpha_poly[ii][ii] = (ll_alpha_poly[ii-1][ii-1] + (ii-1)) % 255
            
        #-----------------------------------------------------------------------
        # Polynomial arithmetic for QR Code is calculated using:
        #     bit-wise modulo 2 arithmetic
        #     byte-wise module 100011101 arithmetic {Galois field of 2^8; prime modulus polynomial (x^8) + (x^4) + (x^3) + (x^2) + (x^0)}
        # EC codewords are the remainder after dividing the data by polynomial g(x).
        #-----------------------------------------------------------------------
                
        m_polynomial = np.append( m_data, np.zeros(i_degree, dtype=np.uint8))
        
        
        for ii in range(m_data.size):
            if m_polynomial[ii] == 0 :
                continue
            m_xor = np.array([ alpha_log_table[(exp + alpha_antilog_table[m_polynomial[ii]]) % 255] for exp in ll_alpha_poly[i_degree] ])
            m_polynomial[ii:(ii+i_degree+1)] = np.bitwise_xor( m_polynomial[ii:(ii+i_degree+1)], m_xor )
            
        return m_polynomial[(0-i_degree):]
        
    def fill_data(self, m_data):
    
        #==============================================================================#
        # Data Placement!
        #   - How does one fill the encoding region with the data + error correction info?
        #   - The rules are fairly rigid and easy to understand.
        #       - "Codewords" are 8 modules.
        #       - Start at bottom right corner of symbol.
        #       - Consider 2 columns at a time, and snake up and down the encoding region, moving left upon hitting top/bottom. (Place module in right first, then left, in 2 column section.)
        #       - Ok if shapes are irregular, IE: if top is reached before a codeword is finished, or if non-encoding region space blocks the way.
        #==============================================================================#

    
        if self.codewords <= (self.blocks * self.ec_per_block) :
            raise Exception("Error correction size larger than # available code words...")
        
        #---------------------------------#
        #  Pad data + split into blocks.  #
        #---------------------------------#
        
        # 'Pad Codewords' xEC11
        m_pad = np.tile( np.array([0xEC, 0x11], dtype=np.uint8),  1 + ((self.codewords - (self.blocks * self.ec_per_block))>>1) )
        m_data_codewords = np.append( m_data, m_pad[:(self.codewords-(self.blocks * self.ec_per_block))-m_data.size:] )

        # Empty EC Codewords
        m_ec_codewords = np.zeros((self.ec_per_block * self.blocks), dtype=np.uint8)

        # Split data into # blocks.
        i_block_size = int(m_data_codewords.size//self.blocks)
        i_remaining  = int(m_data_codewords.size-(i_block_size*self.blocks))
        i_start      = 0
        
        #-------------------------------------------------------#
        #  Calculate Reed-Solomon EC codewords for each block.  #
        #-------------------------------------------------------#
        
        for ii in range(0, self.blocks - i_remaining) :
            m_ec_codewords[(ii*self.ec_per_block):((ii+1)*self.ec_per_block)] = self.get_rs_ec_remainder( m_data_codewords[i_start:(i_start+i_block_size)], self.ec_per_block)
            i_start += (i_block_size)
            
        for ii in range(self.blocks - i_remaining, self.blocks) :
            m_ec_codewords[(ii*self.ec_per_block):((ii+1)*self.ec_per_block)] = self.get_rs_ec_remainder( m_data_codewords[i_start:(i_start+i_block_size+1)], self.ec_per_block)
            i_start += (i_block_size + 1)
        
        #---------------------#
        #  Re-Org Final Data  #
        #---------------------#
        m_blocked_data = np.reshape(m_data_codewords, (self.blocks,-1))
        m_blocked_ec   = np.reshape(m_ec_codewords, (self.blocks,-1))
        
        m_final_data = np.reshape(m_blocked_data, -1, order='F')
        m_final_rs   = np.reshape(m_blocked_ec, -1, order='F')
        m_final      = np.append(m_final_data, m_final_rs)
        
        self.m_final_data = m_data_codewords
        self.m_final_ec   = m_ec_codewords
        self.m_final_qr   = m_final

        #-------------#
        #  Fill Data  #
        #-------------#
        
        #-----------------------------------------------------------------------
        # "Symbol characters are positioned in two-module wide columns commencing 
        #  at the lower right corner of the symbol and running alternatively upwards
        #  and downwards from the right to the left."
        #
        #  UP: 01      DOWN: 67  UP->DOWN: 0123
        #      23            45              45
        #      45            23              67
        #      67(MSB)       01
        #-----------------------------------------------------------------------
        # - self.cell_shaded = np.zeros((self.side_length, self.side_length), dtype=bool)
        # - self.cell_shaded[row][col]
        #-----------------------------------------------------------------------
        
        # Position
        b_up  = True
        i_col = self.side_length-1
        i_row = self.side_length-1

        # Data
        i_bit = 0x100
        i_byte = 0
        
        def next_bool():
            if(i_byte > m_final.size):
                return False
            return ((m_final[i_byte] & i_bit) != 0)
        
        while i_col > 0 :
            
            # Place bits if possible. ------------------------------------------
            for ii in range(2):
                if self.cell_finished[i_row][i_col-ii] == False :
                    i_bit >>= 1
                    if i_bit == 0 :
                        i_bit = 0x80
                        i_byte += 1
                    self.cell_shaded[i_row][i_col-ii] = next_bool()
            
            # Go up/down. ------------------------------------------------------
            if b_up :
                i_row -= 1 # UP
            else :
                i_row += 1 # DOWN
            
            # If past limit, switch direction. ---------------------------------
            if b_up :
                if i_row >= self.side_length :
                    raise Exception("<>.")
                elif i_row < 0 :
                    i_col -= 2
                    if(i_col == 6): # Vertical Alignment
                        i_col = 5
                    i_row = 0
                    b_up = False
            else :
                if i_row < 0 :
                    raise Exception("<>.")
                elif i_row >= self.side_length :
                    i_col -= 2
                    if(i_col == 6): # Vertical Alignment
                        i_col = 5
                    i_row = self.side_length-1
                    b_up = True
        
        return
    
    #===========================#
    #  QR Functions - Non-Data  #
    #===========================#
    
    def fill_finder_patterns(self):
        
        # Top-Left Finder Pattern (& Seperator)
        self.qr_draw_finder((0,0))
        self.qr_draw_row(False, (7,0), 8)
        self.qr_draw_col(False, (0,7), 7)
        
        # Top-Right Finder Pattern (& Seperator)
        self.qr_draw_finder((0,self.side_length-7))
        self.qr_draw_row(False, (7,self.side_length-8), 8)
        self.qr_draw_col(False, (0,self.side_length-8), 7)
        
        # Bottom-Left Finder Pattern (& Seperator)
        self.qr_draw_finder((self.side_length-7,0))
        self.qr_draw_row(False, (self.side_length-8,0), 7)
        self.qr_draw_col(False, (self.side_length-8,7), 8) 

    def fill_timing_patterns(self): # 6.3.5 (Pg 17)
        
        # Timing Row
        for ii in range( 8, self.side_length-8 ):
            row, col = (6, ii)
            self.qr_draw_pixel(((ii % 2) == 0), (row, col))
        
        # Timing Column
        for ii in range( 8, self.side_length-8 ):
            row, col = (ii, 6)
            self.qr_draw_pixel(((ii % 2) == 0), (row, col))
    
    def fill_alignment_patterns(self):
    
        # The 'n' coordinates for the alignment patterns will form a (n^2 - 3) grid.
        # Alignment patterns cannot overlap Finder patterns, hence the 3 near the Finders are not placed.
        # Remove: (min,min), (min,max), (max,min)
        if self.alignment_list is None :
            return
        
        for row in self.alignment_list :
            for col in self.alignment_list :
            
                # Skip (min,min), (min,max), (max,min)
                if (row == self.alignment_list[0]) and ((col == self.alignment_list[0]) or (col == self.alignment_list[-1])) :
                    continue
                if (col == self.alignment_list[0]) and ((row == self.alignment_list[0]) or (row == self.alignment_list[-1])) :
                    continue
                
                # "self.alignment[][]" indicates center square position:
                self.qr_draw_alignment((row-2,col-2))
    
    def fill_version_info(self, qr_ver):
    
        #==============================================================================#
        # Version Info [QR Code Symbol]
        #  == 18-bit (in QR Code v7 or larger)
        #      - 6  == Version # (7 to 40)
        #      - 12 == Error Correction
        #------------------------------------------------------------------------------#
        #  - Twice for redundancy:
        #      - Right of top-right finder, 3x6, lsb in top-left, write left-to-write, down a row each 3 bits.
        #      - Above bot-left finder, 6x3, lsb in top-left, write down, each colunm w/ 3 bits, going left to right.
        #==============================================================================#
        # Convert input data in polynomial, divide by generator polynomial G(x)
        # G(x) = x^12 + x^11 + x^10 + x^9 + x^8 + x^5 + x^2 + x^0  [x1F25 == 7973]
        const_golay = 7973
        #-------------------------------------------------------------------------------------------
        # EG: 7
        #-------------------------------------------------------------------------------------------
        # "000111"
        # (x^2 + x + 1)*(x^12) / (x^12 + x^11 + x^10 + x^9 + x^8 + x^5 + x^2 + x^0)
        #-------------------------------------------------------------------------------------------
        # [17] [16] [15] [14] [13] [12] [11] [10] [ 9] [ 8] [ 7] [ 6] [ 5] [ 4] [ 3] [ 2] [ 1] [ 0]
        #-------------------------------------------------------------------------------------------
        #   0    0    0    1    1    1    0    0    0    0    0    0    0    0    0    0    0    0
        # Divide by G(x) * (x^2)
        #                  1    1    1    1    1    0    0    1    0    0    1    0    1
        #-------------------------------------------------------------------------------------------
        #                  0    0    0   -1   -1    0    0   -1    0    0   -1    0   -1    0    0
        #-------------------------------------------------------------------------------------------
        # Therefore:
        #   (x^2 + x + 1)*(x^12) / (x^12 + x^11 + x^10 + x^9 + x^8 + x^5 + x^2 + x^0)
        #     =
        #   (x^12 + x^11 + x^10 + x^9 + x^8 + x^5 + x^2 + x^0)(x^2) + (x^11 + x^10 + x^7 + x^4 + x^2)
        #-------------------------------------------------------------------------------------------
        # Golay error code := "110010010100"
        #-------------------------------------------------------------------------------------------
        
        # Only encode version info if version 7 or greater...
        if(qr_ver < 7):
            return
        
        #-------------------------------------------------------------------------------------------
        # Max:
        #      11_1111_0000_0000_0000
        #   +        1_1111_0010_0101
        #----------------------------
        #     100_0000_1111_0010_0101
        # vs.  11_1110_0100_101<_<<<<
        #-------------------------------------------------------------------------------------------
        i_poly = (qr_ver << 12)
        for ii in range(5, -1, -1):
            if(i_poly > (const_golay << ii)):
                i_poly = i_poly - (const_golay << ii)
                
        i_poly = (const_golay - i_poly) # Get remainder!
        i_ver_info = (qr_ver << 12) + i_poly
        
        #----------------------------------------------------------------------#
        # Draw Info.
        #----------------------------------------------------------------------#
        for ii in range(3):
            for jj in range(6):
                i_bit = (1<<jj)+ii
                
                # Top-Right; Ver Info
                self.qr_draw_pixel( (((i_ver_info >> i_bit) & 1) == 1), (jj, (self.side_length-11)+ii) )
                
                # Bottom-Right; Ver Info
                self.qr_draw_pixel( (((i_ver_info >> i_bit) & 1) == 1), ((self.side_length-11)+ii, jj) )
    
    #===============================================#
    #  QR Function - Calculate/Choose Best Mask...  #
    #===============================================#
    
    def fill_format_into(self, i_format_xor):
    
        for ii in range(0,6):
            self.qr_draw_pixel( (((i_format_xor >> ii) & 1) == 1), (ii, 8) ) # (Around) Top-Left Finder
            self.qr_draw_pixel( (((i_format_xor >> ii) & 1) == 1), (8, (self.side_length-1)-ii) ) # (Around) Top-Right Finder
        
        for ii in range(6,8):
            self.qr_draw_pixel( (((i_format_xor >> ii) & 1) == 1), ((ii+1), 8) ) # (Around) Top-Left Finder
            self.qr_draw_pixel( (((i_format_xor >> ii) & 1) == 1), (8, (self.side_length-1)-ii) ) # (Around) Top-Right Finder
        
        # Dark Module
        self.qr_draw_pixel( True, (self.side_length-8,8) )
        
        for ii in range(8,9):
            self.qr_draw_pixel( (((i_format_xor >> ii) & 1) == 1), (8, 15-ii) ) # (Around) Top-Left Finder
            self.qr_draw_pixel( (((i_format_xor >> ii) & 1) == 1), ((self.side_length-15)+ii, 8) ) # (Around) Bottom-Left Finder
        
        for ii in range(9,15):
            self.qr_draw_pixel( (((i_format_xor >> ii) & 1) == 1), (8, 14-ii) ) # (Around) Top-Left Finder
            self.qr_draw_pixel( (((i_format_xor >> ii) & 1) == 1), ((self.side_length-15)+ii, 8) ) # (Around) Bottom-Left Finder
    
    def select_mask_fill_format(self, data_mask):
    
        #==============================================================================#
        # Format Information [QR Code Symbol]
        #   == 5 bits + 10 error corrections bits
        #       - 2 Bits == Error Correction Level [M, L, H, Q]
        #       - 3 Bits == Data Mask Pattern
        #   - 10 Error Correction Bits
        #   - All 15 bits XORed w/ 101_0100_0001_0010 (Ensure no combo of data/ec can be all zeros...)
        #------------------------------------------------------------------------------#
        #   - This 15 module code appears twice for redundancy!
        #       (1) From top->down->left, goes around top-left finder pattern.
        #       (2) 8 modules below top right-finder pattern, and right of bot-left finder.
        #==============================================================================#

        #==============================================================================#
        # Format Information [Micro QR Code Symbol] (page 57)
        #  - Wraps around finder/empty space from top->down->left.
        #==============================================================================#
    
        #----------------------------------------------------------------------#
        # Bose-Chaudhuri-Hocquenghem (15,5) Code
        # G(x) = generator polynomial := (x^10) + (x^8) + (x^5) + (x^4) + (x^2) + (x^1) + (x^0)
        #----------------------------------------------------------------------#
        const_bch_poly = 1335 # := 101_0011_0111
        const_qr_xor = 21522 # XOR w/ 101_0100_0001_0010 = 21522
        
        #----------------------------------------------------------------------#
        if(self.err_corr_sel == 'L'):
            i_format_info = (1<<3) + data_mask
        elif(self.err_corr_sel == 'M'):
            i_format_info = (0<<3) + data_mask
        elif(self.err_corr_sel == 'Q'):
            i_format_info = (3<<3) + data_mask
        elif(self.err_corr_sel == 'H'):
            i_format_info = (2<<3) + data_mask
        else:
            raise Exception("<>.")
            
        #----------------------------------------------------------------------#
        i_poly = (i_format_info << 10)
        #----------------------------------------------------------------------#
        # Max:
        #     111_1100_0000_0000
        # vs. 101_0011_0111_<<<<
        #----------------------------------------------------------------------#
        # EG:    DATA :=             0_0010
        #          EC :=       10_0110_1110
        #  w/out MASK := 000_1010_0110_1110
        #     w/ MASK := 101_1110_0111_1100 := x5E7C
        #----------------------------------------------------------------------#
        for ii in range(4, -1, -1):
            if(i_poly >= (const_bch_poly << ii)):
                i_poly = i_poly - (const_bch_poly << ii)
        i_poly = (const_bch_poly - i_poly)
        i_format_xor = ((i_format_info << 10) + i_poly) ^ const_qr_xor
        
        #----------------------------------------------------------------------#
        m_mask = self.cell_shaded.copy()
        
        #----------------------------------------------------------------------#
        #  Add format info.
        #----------------------------------------------------------------------#
        for ii in range(0,6):
            m_mask[ii][8]                      = (((i_format_xor >> ii) & 1) == 1) # (Around) Top-Left Finder
            m_mask[8][(self.side_length-1)-ii] = (((i_format_xor >> ii) & 1) == 1) # (Around) Top-Right Finder
        
        for ii in range(6,8):
            m_mask[ii+1][8]                    = (((i_format_xor >> ii) & 1) == 1) # (Around) Top-Left Finder
            m_mask[8][(self.side_length-1)-ii] = (((i_format_xor >> ii) & 1) == 1) # (Around) Top-Right Finder
        
        m_mask[self.side_length-8][8] = True # Dark Module
        
        for ii in range(8,9):
            m_mask[8][15-ii]                    = (((i_format_xor >> ii) & 1) == 1) # (Around) Top-Left Finder
            m_mask[(self.side_length-15)+ii][8] = (((i_format_xor >> ii) & 1) == 1) # (Around) Bottom-Left Finder
        
        for ii in range(9,15):
            m_mask[8][14-ii]                    = (((i_format_xor >> ii) & 1) == 1) # (Around) Top-Left Finder
            m_mask[(self.side_length-15)+ii][8] = (((i_format_xor >> ii) & 1) == 1) # (Around) Bottom-Left Finder
        
        #==============================================================================#
        # Data Masking
        #   - Balance light and dark.
        #   - Avoid finder pattern!!! [1011101]
        #       - xxxxxxx
        #       - xooooox
        #       - xoxxxox
        #       - xoxxxox
        #       - xoxxxox
        #       - xooooox
        #       - xxxxxxx
        #   - Data masking not applied to function patterns!  Encoding region (excluding format/version info).
        #   - Select pattern with lowest # penalties...
        #==============================================================================#
        
        #=======================================================================================================
        # Available patterns: 'row' / 'col', where r=0 and c=0 is the top left module.
        #  --------------+--------------------+---------------------------------------+-------
        #   Mask QR Code | Mask Micro QR Code | Style                                 | Desc.
        #  --------------+--------------------+---------------------------------------+-------
        #            000 |                    | (r+c)       mod 2 = 0                 | Checkerboard.
        #            001 |                 00 | (r)         mod 2 = 0                 | Every other row.
        #            010 |                    | (c)         mod 3 = 0                 | Every 3rd column.
        #            011 |                    | (r+c)       mod 3 = 0                 | Every 3rd diagonal.
        #            100 |                 01 | (r/2)+(c/3) mod 2 = 0                 | Fatter checkerboard.
        #  --------------+--------------------+---------------------------------------+-------
        #            101 |                    | ((r c) mod 2 + (r c) mod 3)       = 0 | Plus symbols in boxes.
        #            110 |                 10 | ((r c) mod 2 + (r c) mod 3) mod 2 = 0 | Diamond-ish symbols!
        #            111 |                 11 | ((r+c) mod 2 + (r c) mod 3) mod 2 = 0 | Diagonal rectangles...
        #=======================================================================================================
        for r in range(self.side_length):
            for c in range(self.side_length):
                if self.cell_finished[r][c] :
                    continue
                match data_mask:
                    case 0:
                        if( (r+c)%2 == 0 ):
                            m_mask[r][c] = not m_mask[r][c]
                    case 1:
                        if( (r)%2 == 0 ):
                            m_mask[r][c] = not m_mask[r][c]
                    case 2:
                        if( (c)%3 == 0 ):
                            m_mask[r][c] = not m_mask[r][c]
                    case 3:
                        if( (r+c)%3 == 0 ):
                            m_mask[r][c] = not m_mask[r][c]
                    case 4:
                        if( ((r//2)+(c//3))%2 == 0 ):
                            m_mask[r][c] = not m_mask[r][c]
                    case 5:
                        if( (r*c)%2 + (r*c)%3 == 0 ):
                            m_mask[r][c] = not m_mask[r][c]
                    case 6:
                        if( (((r*c)%2) + ((r*c)%3))%2 == 0 ):
                            m_mask[r][c] = not m_mask[r][c]
                    case 7:
                        if( (((r+c)%2) + ((r*c)%3))%2 == 0 ):
                            m_mask[r][c] = not m_mask[r][c]
        return m_mask

#------------------------------------------------------------------------------#

class QR_Code_Gen :
    
    #------------------#
    #  Initialization  #
    #------------------#
    
    def __init__(self):
    
        # #unused bits.
        self.v_to_remainder     = [None, 0, 7,7,7,7,7, 0,0,0,0,0,0,0, 3,3,3,3,3,3,3, 4,4,4,4,4,4,4, 3,3,3,3,3,3,3, 0,0,0,0,0,0]
        
        # #codewords in each QR version, and #codewords needed for error-correction.
        self.v_to_num_codewords = [None, 26, 44, 70, 100, 134, 172, 196, 242, 292, 346, 404, 466, 532, 581, 655, 733, 815, 901, 991, 1085, 1156, 1258, 1364, 1474, 1588, 1706, 1828, 1921, 2051, 2185, 2323, 2465, 2611, 2761, 2876, 3034, 3196, 3362, 3532, 3706]
        self.v_to_ec_codewords = [None,{'L':7,'M':10,'Q':13,'H':17},{'L':10,'M':16,'Q':22,'H':28},{'L':15,'M':26,'Q':36,'H':44},{'L':20,'M':36,'Q':52,'H':64},{'L':26,'M':48,'Q':72,'H':88},{'L':36,'M':64,'Q':96,'H':112},{'L':40,'M':72,'Q':108,'H':130},{'L':48,'M':88,'Q':132,'H':156},{'L':60,'M':110,'Q':160,'H':192},{'L':72,'M':130,'Q':192,'H':224},{'L':80,'M':150,'Q':224,'H':264},{'L':96,'M':176,'Q':260,'H':308},{'L':104,'M':198,'Q':288,'H':352},{'L':120,'M':216,'Q':320,'H':384},{'L':132,'M':240,'Q':360,'H':432},{'L':144,'M':280,'Q':408,'H':480},{'L':168,'M':308,'Q':448,'H':532},{'L':180,'M':338,'Q':504,'H':588},{'L':196,'M':364,'Q':546,'H':650},{'L':224,'M':416,'Q':600,'H':700},{'L':224,'M':442,'Q':644,'H':750},{'L':252,'M':476,'Q':690,'H':816},{'L':270,'M':504,'Q':750,'H':900},{'L':300,'M':560,'Q':810,'H':960},{'L':312,'M':588,'Q':870,'H':1050},{'L':336,'M':644,'Q':952,'H':1110},{'L':360,'M':700,'Q':1020,'H':1200},{'L':390,'M':728,'Q':1050,'H':1260},{'L':420,'M':784,'Q':1140,'H':1350},{'L':450,'M':812,'Q':1200,'H':1440},{'L':480,'M':868,'Q':1290,'H':1530},{'L':510,'M':924,'Q':1350,'H':1620},{'L':540,'M':980,'Q':1440,'H':1710},{'L':570,'M':1036,'Q':1530,'H':1800},{'L':570,'M':1064,'Q':1590,'H':1890},{'L':600,'M':1120,'Q':1680,'H':1980},{'L':630,'M':1204,'Q':1770,'H':2100},{'L':660,'M':1260,'Q':1860,'H':2220},{'L':720,'M':1316,'Q':1950,'H':2310},{'L':750,'M':1372,'Q':2040,'H':2430}]
        
        # #Block of data/codewords per version & security level.
        self.v_to_blocks = [None,{'L':1,'M':1,'Q':1,'H':1},{'L':1,'M':1,'Q':1,'H':1},{'L':1,'M':1,'Q':2,'H':2},{'L':1,'M':2,'Q':2,'H':4},{'L':1,'M':2,'Q':4,'H':4},{'L':2,'M':4,'Q':4,'H':4},{'L':2,'M':4,'Q':6,'H':5},{'L':2,'M':4,'Q':6,'H':6},{'L':2,'M':5,'Q':8,'H':8},{'L':4,'M':5,'Q':8,'H':8},{'L':4,'M':5,'Q':8,'H':11},{'L':4,'M':8,'Q':10,'H':11},{'L':4,'M':9,'Q':12,'H':16},{'L':4,'M':9,'Q':16,'H':16},{'L':6,'M':10,'Q':12,'H':18},{'L':6,'M':10,'Q':17,'H':16},{'L':6,'M':11,'Q':16,'H':19},{'L':6,'M':13,'Q':18,'H':21},{'L':7,'M':14,'Q':21,'H':25},{'L':8,'M':16,'Q':20,'H':25},{'L':8,'M':17,'Q':23,'H':25},{'L':9,'M':17,'Q':23,'H':34},{'L':9,'M':18,'Q':25,'H':30},{'L':10,'M':20,'Q':27,'H':32},{'L':12,'M':21,'Q':29,'H':35},{'L':12,'M':23,'Q':34,'H':37},{'L':12,'M':25,'Q':34,'H':40},{'L':13,'M':26,'Q':35,'H':42},{'L':14,'M':28,'Q':38,'H':45},{'L':15,'M':29,'Q':40,'H':48},{'L':16,'M':31,'Q':43,'H':51},{'L':17,'M':33,'Q':45,'H':54},{'L':18,'M':35,'Q':48,'H':57},{'L':19,'M':37,'Q':51,'H':60},{'L':19,'M':38,'Q':53,'H':63},{'L':20,'M':40,'Q':56,'H':66},{'L':21,'M':43,'Q':59,'H':70},{'L':22,'M':45,'Q':62,'H':74},{'L':24,'M':47,'Q':65,'H':77},{'L':25,'M':49,'Q':68,'H':81}]
        self.v_to_ec_per_block = [None,{'L':7,'M':10,'Q':13,'H':17},{'L':10,'M':16,'Q':22,'H':28},{'L':15,'M':26,'Q':18,'H':22},{'L':20,'M':18,'Q':26,'H':16},{'L':26,'M':24,'Q':18,'H':22},{'L':18,'M':16,'Q':24,'H':28},{'L':20,'M':18,'Q':18,'H':26},{'L':24,'M':22,'Q':22,'H':26},{'L':30,'M':22,'Q':20,'H':24},{'L':18,'M':26,'Q':24,'H':28},{'L':20,'M':30,'Q':28,'H':24},{'L':24,'M':22,'Q':26,'H':28},{'L':26,'M':22,'Q':24,'H':22},{'L':30,'M':24,'Q':20,'H':24},{'L':22,'M':24,'Q':30,'H':24},{'L':24,'M':28,'Q':24,'H':30},{'L':28,'M':28,'Q':28,'H':28},{'L':30,'M':26,'Q':28,'H':28},{'L':28,'M':26,'Q':26,'H':26},{'L':28,'M':26,'Q':30,'H':28},{'L':28,'M':26,'Q':28,'H':30},{'L':28,'M':28,'Q':30,'H':24},{'L':30,'M':28,'Q':30,'H':30},{'L':30,'M':28,'Q':30,'H':30},{'L':26,'M':28,'Q':30,'H':30},{'L':28,'M':28,'Q':28,'H':30},{'L':30,'M':28,'Q':30,'H':30},{'L':30,'M':28,'Q':30,'H':30},{'L':30,'M':28,'Q':30,'H':30},{'L':30,'M':28,'Q':30,'H':30},{'L':30,'M':28,'Q':30,'H':30},{'L':30,'M':28,'Q':30,'H':30},{'L':30,'M':28,'Q':30,'H':30},{'L':30,'M':28,'Q':30,'H':30},{'L':30,'M':28,'Q':30,'H':30},{'L':30,'M':28,'Q':30,'H':30},{'L':30,'M':28,'Q':30,'H':30},{'L':30,'M':28,'Q':30,'H':30},{'L':30,'M':28,'Q':30,'H':30},{'L':30,'M':28,'Q':30,'H':30}]
        
        # Coordinates of both x/y alignment patterns.  (QR code will have (N^2)-2 alignment patterns)
        self.v_to_alignment = [None,None,(6,18),(6,22),(6,26),(6,30),(6,34),(6,22,38),(6,24,42),(6,26,46),(6,28,50),(6,30,54),(6,32,58),(6,34,62),(6,26,46,66),(6,26,48,70),(6,26,50,74),(6,30,54,78),(6,30,56,82),(6,30,58,86),(6,34,62,90),(6,28,50,72,94),(6,26,50,74,98),(6,30,54,78,102),(6,28,54,80,106),(6,32,58,84,110),(6,30,58,86,114),(6,34,62,90,118),(6,26,50,74,98,122),(6,30,54,78,102,126),(6,26,52,78,104,130),(6,30,56,82,108,134),(6,34,60,86,112,138),(6,30,58,86,114,142),(6,34,62,90,118,146),(6,30,54,78,102,126,150),(6,24,50,76,102,128,154),(6,28,54,80,106,132,158),(6,32,58,84,110,136,162),(6,26,54,82,110,138,166),(6,30,58,86,114,142,170)]
    
    #------------------------#
    #  Test/Print Functions  #
    #------------------------#
    
    def print_table_info(self):
        for ii in range(1,41):
            # Print!
            print(  '[', ii, ']  ', self.v_to_ec_codewords[ii], '  =  ', self.v_to_blocks[ii], '  .*  ', self.v_to_ec_per_block[ii] )
            # Test!
            for rs_level in ['L','M','Q','H'] :
                if  self.v_to_ec_codewords[ii][rs_level] !=  (self.v_to_blocks[ii][rs_level] * self.v_to_ec_per_block[ii][rs_level]) :
                    print("\t[",str(ii),"][",rs_level,"]  ::  ",self.v_to_ec_codewords[ii][rs_level],"  !=  ",self.v_to_blocks[ii][rs_level],"  *  ",self.v_to_ec_per_block[ii][rs_level])
    
    #--------------#
    #  Parse Data  #
    #--------------#
    
    def parse_numeric_encoding(self, in_str):
        t_str = in_str
        i_data = 0
        i_len  = 0
        
        # Convert # string into groups of 3-decimal-digits, starting with msb...
        while len(t_str) > 3:
            t_data, t_str = t_str[:3], t_str[3:]
            i_data = (i_data << 10) + int(t_data)
            i_len += 10
            
        # Remaining 1/2/3 #s.
        match len(t_str):
            case 3:
                i_data = (i_data << 10) + int(t_str)
                i_len += 10
            case 2:
                i_data = (i_data << 7) + int(t_str)
                i_len += 7
            case 1:
                i_data = (i_data << 4) + int(t_str)
                i_len += 4
            case _:
                raise Exception("Length of remaining string = "+str(len(t_str))+" is unexpected.")
        
        return (i_data, i_len)
    
    def parse_alphanumeric_encoding(self, in_str): # 0-9 A-Z $ % * + - . / : (Space)
        t_str = in_str
        i_data = 0
        i_len  = 0
        
        #----------------------------------------------------------------------#
        # Encoding Table
        #----------------------------------------------------------------------#
        #    0-9 |  0 -  9
        #    A-Z | 10 - 35
        #     SP | 36
        #   #%*+ | 37 - 40
        #      - | 41
        #    ./: | 42 - 44
        # Input characters divided into groups of 2, (45*y)+z, 11-bits; final odd character is 6-bits.
        #----------------------------------------------------------------------#
        def parse_alphanumeric_char_convert(in_char):
            i_char = ord(in_char)
            match i_char:
                case i_num if ord('0') <= i_num <= ord('9'): # x30 - x39
                    return i_char - ord('0')
                case i_num if ord('A') <= i_num <= ord('Z'): # x41 - 5A
                    return i_char - ord('A')
                case ord(' '): # x20
                    return 36
                case ord('#'): # x23
                    return 37
                case ord('%'): # x25
                    return 38
                case ord('*'): # x2A
                    return 39
                case ord('+'): # x2B
                    return 40
                case ord('-'): # x2D
                    return 41
                case ord('.'): # x2E
                    return 42
                case ord('/'): # x2F
                    return 43
                case ord(':'): # x3A
                    return 44
                case _:
                    raise Exception("Alphanumeric char["+str(in_char)+"] is unexpected.")
        
        while len(t_str) > 2:
            c_1, c_2, t_str = t_str[0], t_str[1], t_str[2:]
            i_1 = parse_alphanumeric_char_convert(c_1)
            i_2 = parse_alphanumeric_char_convert(c_2)
            i_data = (i_data << 11) + (i_1 * 45) + (i_2)
            i_len += 11
            
        match len(t_str):
            case 2:
                i_1 = parse_alphanumeric_char_convert(t_str[0])
                i_2 = parse_alphanumeric_char_convert(t_str[1])
                i_data = (i_data << 11) + (i_1 * 45) + (i_2)
                i_len += 11
            case 1:
                i_1 = parse_alphanumeric_char_convert(t_str[0])
                i_data = (i_data << 6) + (i_1)
                i_len += 6
            case _:
                raise Exception("Length of remaining string = "+str(len(t_str))+" is unexpected.")
        
        return (i_data, i_len)
    
    def parse_byte_encoding(self, in_str):
        return (int.from_bytes(bytes(in_str,"utf-8"),"big"), len(in_str)*8)
    
    #------------------------------#
    #  QR Generator Vanilla Modes  #
    #------------------------------#
    def gen_QR_numeric_encoding(self, in_str, ec='Q'): # Just #s!
    
        #----------------------------------------------------------------------#
        # TODO: Regular Expression check...
        #----------------------------------------------------------------------#
        
        #----------------------------------------------------------------------#
        # Parse Numeric Data
        #----------------------------------------------------------------------#
        (i_data, i_count) = self.parse_numeric_encoding(in_str)
        
        #----------------------------------------------------------------------#
        # Numeric Mode, Char Count Bits:  [V1-V9]=10 [V10-V26]=12 [V27-V40]=14
        # (Segment + Terminator) = (Mode + Char Count + Data + Terminator)
        #----------------------------------------------------------------------#
        for ii in range(1,10):
            
            if (self.v_to_num_codewords[ii] - self.v_to_ec_codewords[ii][ec]) >= int( np.ceil((4 + 10 + i_count + 4)/8) ) :
                           
                # Construct segment.
                t_mode = 1
                t_segment = ((((t_mode<<10) + len(in_str))<<i_count) + i_data)<<4
                t_len = 4 + 10 + i_count + 4
                
                # Pad with 0s if not a multiple of 8.
                t_segment <<= (8 - (t_len % 8))
                
                # Convert to numpy-array.
                segment_matrix = np.zeros( np.ceil(t_len/8).astype(int), dtype=np.uint8)
                for jj in range( np.ceil(t_len/8).astype(int)-1, -1, -1):
                    segment_matrix[jj] = (t_segment >> (jj<<3)) % (1<<8)
                segment_matrix = np.flip(segment_matrix)
                                
                # Return QR Code
                return QR_Code(                                       \
                    i_qr_ver        = ii,                             \
                    c_ec            = ec,                             \
                    m_data          = segment_matrix,                 \
                    i_num_codewords = self.v_to_num_codewords[ii],    \
                    i_num_blocks    = self.v_to_blocks[ii][ec],       \
                    i_ec_per_block  = self.v_to_ec_per_block[ii][ec], \
                    l_alignment     = self.v_to_alignment[ii]         \
                )
            
        for ii in range(10,27):
            return QR_Code()
        
        for ii in range(27,41):
            return QR_Code()
        
        raise Exception("Number too large to be represented by QR code V40.")
        
    def gen_QR_alphanumeric_encoding(self, in_str): # 0-9 A-Z $ % * + - . / : (Space)
        return QR_Code()
    
    def gen_QR_byte_encoding(self, in_str): # 8-bits per character.
        return QR_Code()
        
    # def gen_QR_kanji_encoding(self, in_str):
        # return QR_Code()
        
    # def gen_ECI(self, in_str):
    #==============================================================================#
    # ECI Header:
    #   - (4 bits)       ECI Mode Indicator
    #   - (8/16/24 bits) ECI Designator
    #       - 6 digit assignment [Mindfuck, they mean "6 decimal human digits"]
    #           - (decimal) d000000 - d000127 --> (binary) 0xxxxxxx
    #           - (decimal) d000128 - d016383 --> (binary) 10xxxxxx xxxxxxxx
    #           - (decimal) d016384 -         --> (binary) 110xxxxx xxxxxxxx xxxxxxxx
    #
    # In default ECI, bit stream commences with the first mode indicator.  If 
    #   anything other than default ECI, bit stream commences with header, followed
    #   by 1st segment.
    #
    # Remainder of bit stream is:
    #   - Mode indicator
    #   - Character count
    #   - Data bit stream
    #
    # Final bit stream (example) :=
    #   - (ECI 4bit)(ECI Assignment 8bit)(Mode Indicator 'byte mode' 4bit)(Character Count 8bit)(data 5bytes)
    #
    # Some shenanigans with the character x5C
    #==============================================================================#
        
    #---------------------------------#
    #  QR Generator Additional Modes  #
    #---------------------------------#
        
    #----------------------#
    #  Special QR Formats  #
    #----------------------#
    
    # FNC1 (?)
    
    # Mixing
    
    # Structured Append
    
    #--------------------------#
    #  Catch-all QR Generator  #
    #--------------------------#
    def gen_QR(self, in_str):
    
        # Error if not string.
        if not isinstance(in_str, str):
            raise Exception("Expecting input type 'str'.")
        
        # Decode data:
        data = None
        
        # Determine how big QR needed based on data.
        necessary_codewords = None
        ver = 1
        while( necessary_codewords > self.v_to_num_codewords[ii] ):
            ver += 1
            if ver > 40 :
                raise Exception("Byte data exceeds QR Code v40 size.")
        
        # Generate QR Code class.
        return QR_Code(ver, data)

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

    #--------------------------------#
    #  "QR Code Generator" Instance  #
    #--------------------------------#
    qr_gen_test      = QR_Code_Gen()
    qr_code_01234567 = qr_gen_test.gen_QR_numeric_encoding('01234567', ec='M')