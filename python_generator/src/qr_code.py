import numpy as np

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

class QR_Code :

    def __init__(self, i_qr_ver, c_ec, m_data, i_num_codewords, i_num_blocks, i_ec_per_block, l_alignment):
        
        # Version and Format Codes...
        self.ver_code = [None, None, None, None, None, None, None, 0x07C94, 0x085BC, 0x09A99, 0xA4D3, 0x0BBF6, 0x0C762, 0x0D847, 0x0E60D, 0x0F928, 0x10B78, 0x1145D, 0x12A17, 0x13532, 0x149A6, 0x15683, 0x168C9, 0x177EC, 0x18EC4, 0x191E1, 0x1AFAB, 0x1B08E, 0x1CC1A, 0x1D33F, 0x1ED75, 0x1F250, 0x209D5, 0x216F0, 0x228BA, 0x2379F, 0x24B0B, 0x2542E, 0x26A64, 0x27541, 0x28C69]
        self.format_code = [0x5412, 0x5125, 0x5E7C, 0x5B4B, 0x45F9, 0x40CE, 0x4F97, 0x4AA0, 0x77C4, 0x72F3, 0x7DAA, 0x789D, 0x662F, 0x6318, 0x6C41, 0x6976, 0x1689, 0x13BE, 0x1CE7, 0x19D0, 0x0762, 0x0255, 0x0D0C, 0x083B, 0x355F, 0x3068, 0x3F31, 0x3A06, 0x24B4, 0x2183, 0x2EDA, 0x2BED]
        
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
        
        self.m_final_data = None
        self.m_final_ec   = None
        self.m_final_qr   = None
        
        # https://numpy.org/doc/stable/reference/routines.matlib.html
        self.cell_finished = np.zeros((self.side_length, self.side_length), dtype=bool)
        self.cell_shaded   = np.zeros((self.side_length, self.side_length), dtype=bool)
        
        # Function Patterns
        self.fill_finder_patterns()    #
        self.fill_timing_patterns()    # Oscillating strips that run between finder patterns.
        self.fill_alignment_patterns() # Places: [5x5 dark, 3x3 light, 1 dark center]
        
        # Info
        self.fill_version_info(i_qr_ver)
        self.fill_format_into(0) # Pre-emptive claim of format info == '0' ...
        
        # Data + Final Mask
        self.fill_data(m_data)
        
        self.cell_list_masked = [None]*8
        self.cell_list_score  = [None]*8
        for ii in range(8):
            self.cell_list_masked[ii] = self.select_mask_fill_format(ii) # Overwrites the '0' format info.
            self.cell_list_score[ii] = self.qr_code_penalty(self.cell_list_masked[ii])
            
        
        self.best_qr = 0
        min_score    = sum(self.cell_list_score[0])
        for ii in range(1,8):
            if(sum(self.cell_list_score[ii]) < min_score):
                self.best_qr = ii
                min_score    = sum(self.cell_list_score[ii])
    
    #==================#
    #  Test Functions  #
    #==================#
    
    def get_string_info(self):
    
        t_str = ""
        
        t_str += "\nVersion #                : "+str(self.version)
        t_str += "\nReed-Solomon EC Level    : "+str(self.err_corr_sel)
        t_str += "\n# Codewords in QR        : "+str(self.codewords)
        t_str += "\n# Blocks in QR           : "+str(self.blocks)
        t_str += "\n# EC Codewords per Block : "+str(self.ec_per_block)
        t_str += "\n# Alignment List         : "+str(self.alignment_list)
        
        if self.m_final_data is not None :
            t_str += "\n\nData Matrix:"+str(self.m_final_data)
            
        if self.m_final_ec is not None :
            t_str += "\n\nEC Matrix:"+str(self.m_final_ec)
            
        if self.m_final_qr is not None :
            t_str += "\n\nFinal Matrix:"
            for ii in range(self.m_final_qr.size) :
                t_str += "\n\t" + hex(self.m_final_qr[ii])
            
        return t_str
        
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
        
        #-------------------------------------------#
        #  Pre-Calculate Alpha Log/Antilog Table!!  #
        #-------------------------------------------#
        
        i_alpha = 1
        for ii in range(256):
            
            alpha_log_table[ii] = i_alpha
            alpha_antilog_table[i_alpha] = ii
            
            if(i_alpha > 127):
                i_alpha = (i_alpha << 1) ^ 0x11D # prime modulus polynomial (x^8) + (x^4) + (x^3) + (x^2) + (x^0)
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
    
        if self.codewords <= (self.blocks * self.ec_per_block) :
            raise Exception("Error correction size larger than # available code words...")
        
        #---------------------------------#
        #  Pad data + split into blocks.  #
        #---------------------------------#
        
        # 'Pad Codewords' xEC11
        m_pad = np.tile( np.array([0xEC, 0x11], dtype=np.uint8),  1 + ((self.codewords - (self.blocks * self.ec_per_block))>>1) )
        m_data_codewords = np.append( m_data, m_pad[:(self.codewords-(self.blocks * self.ec_per_block))-m_data.size:] )

        # Empty EC Codewords
        m_ec_codewords = np.empty((self.ec_per_block * self.blocks), dtype=np.uint8)

        # Split data into # blocks.
        i_block_min  = int(m_data_codewords.size//self.blocks)
        i_block_max  = int(np.ceil(m_data_codewords.size/self.blocks))
        i_remaining  = int(m_data_codewords.size-(i_block_min*self.blocks)) # How many of the 'blocks' will have '1-extra byte'.
        i_zero_pad   = (0-i_remaining)%self.blocks
        
        if i_zero_pad is not 0:
        
            m_blocked_data = np.empty((self.blocks, i_block_max), dtype=np.uint8)
            
            # 1st-Half (w/ 0 padding).
            m_blocked_data[0:i_zero_pad, 0:i_block_min] = np.reshape(m_data_codewords[:(i_zero_pad*i_block_min)], (i_zero_pad,i_block_min))
            m_blocked_data[0:i_zero_pad, i_block_min] = np.zeros(i_zero_pad, dtype=np.uint8)
            
            # 2nd-Half (data-only).
            m_blocked_data[i_zero_pad:self.blocks,:] = np.reshape(m_data_codewords[(i_zero_pad*i_block_min):], (i_remaining,i_block_max))
            
            # Remove 0-pad...
            m_final_data = np.reshape(m_blocked_data, -1, order='F')
            m_final_data = np.delete(m_final_data, np.arange((self.blocks*i_block_min), (self.blocks*i_block_min)+i_zero_pad) )
            
        else :
            
            m_blocked_data = np.reshape(m_data_codewords, (self.blocks,-1))
            m_final_data = np.reshape(m_blocked_data, -1, order='F')
        
        #-------------------------------------------------------#
        #  Calculate Reed-Solomon EC codewords for each block.  #
        #-------------------------------------------------------#
        
        i_start = 0
        
        for ii in range(0, i_zero_pad) :
            m_ec_codewords[(ii*self.ec_per_block):((ii+1)*self.ec_per_block)] = self.get_rs_ec_remainder( m_data_codewords[i_start:(i_start+i_block_min)], self.ec_per_block)
            i_start += (i_block_min)
            
        for ii in range(i_zero_pad, self.blocks) :
            m_ec_codewords[(ii*self.ec_per_block):((ii+1)*self.ec_per_block)] = self.get_rs_ec_remainder( m_data_codewords[i_start:(i_start+i_block_max)], self.ec_per_block)
            i_start += (i_block_max)
            
        m_blocked_ec = np.reshape(m_ec_codewords, (self.blocks,-1))
        m_final_rs   = np.reshape(m_blocked_ec, -1, order='F')
        
        #---------------------#
        #  Re-Org Final Data  #
        #---------------------#
        m_final = np.append(m_final_data, m_final_rs)
        
        self.m_final_data = m_data_codewords
        self.m_final_ec   = m_ec_codewords
        self.m_final_qr   = m_final

        #==============================================================================#
        # Data Placement!
        #   - How does one fill the encoding region with the data + error correction info?
        #   - "The rules are fairly rigid and easy to understand."
        #       - "Codewords" are 8 modules.
        #       - Start at bottom right corner of symbol.
        #       - Consider 2 columns at a time, and snake up and down the encoding region, moving left upon hitting top/bottom. (Place module in right first, then left, in 2 column section.)
        #       - Ok if shapes are irregular, IE: if top is reached before a codeword is finished, or if non-encoding region space blocks the way.
        #==============================================================================#
        
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
    
        # Only encode version info if version 7 or greater...
        if(qr_ver < 7):
            return
    
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
        i_poly = (qr_ver << 12)
        for ii in range(5, -1, -1):
            if (i_poly>>(ii+12)) != 0 :
                i_poly ^= const_golay << ii
        
        #    if(i_poly > (const_golay << ii)):
        #        i_poly = i_poly - (const_golay << ii) 
        #i_poly = (const_golay - i_poly) # Get remainder!
        
        i_ver_info = (qr_ver << 12) + i_poly
        
        if(i_ver_info != self.ver_code[qr_ver]):
            raise Exception("Version code generated("+hex(i_ver_info)+") does not match expected("+hex(self.ver_code[qr_ver])+") for QR version "+str(qr_ver)+".")
        
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
                
    def qr_code_penalty(self, m_test):
    
        #==============================================================================#
        # Data Scoring for regular QR code:
        #   - Penalty points are as follows: (Weighted penalties: N1 = 3, N2 = 3, N3 = 40, N4 = 10)
        #       (N1) Adjacent modules in both row/col, 5 or more, in same color.
        #           - Points: N1 + (# - 5)
        #       (N2) Each 2x2 block of same color. (3x3 block has 4 2x2 blocks!)
        #           - Points: N2
        #       (N3) The finder pattern, any oscillating 1:1:3:1:1 pattern, before or after a 4 white modules.
        #           - Points: N3
        #       (N4) Proportion of light to dark. (Each 5% in a either direction adds N4 penalty.)
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
    
        m_int = m_test.astype(int)
        m_int8 = m_test.astype(np.int8)
        
        #----------------------------------------------------------------------#
        #  N2 - Find all 2x2 instances.
        #     - Easy to detect back-to-back instances. (Matrix - Shift Matrix) := Change/Stasis...
        #----------------------------------------------------------------------#
        test_n2 = np.zeros((self.side_length-1, self.side_length-1), dtype=np.int8)
        test_n2[:,:] = np.sum( np.array([ m_int8[0:-1,0:-1], m_int8[1:,0:-1], m_int8[0:-1,1:], m_int8[1:,1:] ]), axis=0, dtype=np.int8)
        
        penalty_n2 = 3 * int(np.sum(test_n2 & 0x3 == 0))
        
        #----------------------------------------------------------------------#
        #  N3 - Finder Pattern
        #     - Similar to above, easy to detect 1 pattern in a row, because can just do matrix math operations...
        #----------------------------------------------------------------------#
        #     - 4W:1B:1W:3B:1W:1B
        #     -    1B:1W:3B:1W:1B:4W
        #----------------------------------------------------------------------#
        
        #-------------#
        #  Along Row  #
        #-------------#
        
        m_int8_c_pad = np.concatenate( (np.zeros((self.side_length, 4), dtype=np.int8), m_int8, np.zeros((self.side_length, 4), dtype=np.int8) ), axis=1)
        test_n3ra = np.zeros((self.side_length, self.side_length-6), dtype=np.int8)
        test_n3rb = np.zeros((self.side_length, self.side_length-6), dtype=np.int8)
                      
        test_n3ra[:,:] = np.sum( np.array([ m_int8_c_pad[:,4:-10], m_int8_c_pad[:,6:-8], m_int8_c_pad[:,7:-7], m_int8_c_pad[:,8:-6], m_int8_c_pad[:,10:-4] ]),                       axis=0, dtype=np.int8) \
                       - np.sum( np.array([ m_int8_c_pad[:,5:-9], m_int8_c_pad[:,9:-5], m_int8_c_pad[:,11:-3], m_int8_c_pad[:,12:-2], m_int8_c_pad[:,13:-1], m_int8_c_pad[:,14:] ]), axis=0, dtype=np.int8)
        
        test_n3rb[:,:] = np.sum( np.array([ m_int8_c_pad[:,4:-10], m_int8_c_pad[:,6:-8], m_int8_c_pad[:,7:-7], m_int8_c_pad[:,8:-6], m_int8_c_pad[:,10:-4] ]),                         axis=0, dtype=np.int8) \
                       - np.sum( np.array([ m_int8_c_pad[:,0:-14], m_int8_c_pad[:,1:-13], m_int8_c_pad[:,2:-12], m_int8_c_pad[:,3:-11], m_int8_c_pad[:,5:-9], m_int8_c_pad[:,9:-5] ]), axis=0, dtype=np.int8)
        
        penalty_n3_r = 40 * int(np.sum(test_n3ra == 5)) + int(np.sum(test_n3rb == 5))
        
        #----------------#
        #  Along Column  #
        #----------------#
        
        m_int8_r_pad = np.concatenate( (np.zeros((4, self.side_length), dtype=np.int8), m_int8, np.zeros((4, self.side_length), dtype=np.int8) ), axis=0)
        test_n3ca = np.zeros((self.side_length-6, self.side_length), dtype=np.int8)
        test_n3cb = np.zeros((self.side_length-6, self.side_length), dtype=np.int8)
        
        test_n3ca[:,:] = np.sum( np.array([ m_int8_r_pad[4:-10,:], m_int8_r_pad[6:-8,:], m_int8_r_pad[7:-7,:], m_int8_r_pad[8:-6,:], m_int8_r_pad[10:-4,:] ]),                       axis=0, dtype=np.int8) \
                       - np.sum( np.array([ m_int8_r_pad[5:-9,:], m_int8_r_pad[9:-5,:], m_int8_r_pad[11:-3,:], m_int8_r_pad[12:-2,:], m_int8_r_pad[13:-1,:], m_int8_r_pad[14:,:] ]), axis=0, dtype=np.int8)
        
        test_n3cb[:,:] = np.sum( np.array([ m_int8_r_pad[4:-10,:], m_int8_r_pad[6:-8,:], m_int8_r_pad[7:-7,:], m_int8_r_pad[8:-6,:], m_int8_r_pad[10:-4,:] ]),                         axis=0, dtype=np.int8) \
                       - np.sum( np.array([ m_int8_r_pad[0:-14,:], m_int8_r_pad[1:-13,:], m_int8_r_pad[2:-12,:], m_int8_r_pad[3:-11,:], m_int8_r_pad[5:-9,:], m_int8_r_pad[9:-5,:] ]), axis=0, dtype=np.int8)
        
        penalty_n3_c = 40 * int(np.sum(test_n3ca == 5)) + int(np.sum(test_n3cb == 5))
        
        #----------------------------------------------------------------------#
        #  N4 - Proportion Light to Dark  
        #     - Sum matrix, detect porportion, easy...
        #----------------------------------------------------------------------#
        f_percentage = (100.0 * np.sum(m_int)) / (self.side_length * self.side_length)
        penalty_n4 = int(np.absolute(f_percentage - 50.0) / 5.0) * 10
        
        #----------------------------------------------------------------------#
        #  N1 - Detect block length of same color.
        #     - Detect run length of 5, increment run length detection to 6, etc...
        #     - Continue until run length detection is 0 (or run out of matrix space)
        #     - Add
        #----------------------------------------------------------------------#
    
        #-------------#
        #  Along Row  #
        #-------------#
    
        ii = 5
        runlength_r = list()
        while True:
        
            m_run = np.zeros((self.side_length, self.side_length-(ii-1)), dtype=np.int8)
            for jj in range(ii):
                m_run += m_int8[:,jj:jj+self.side_length-(ii-1)]
            
            run = int(np.sum(m_run == ii))
            if(run == 0):
                break # Completed...
            
            runlength_r.append(run)
            ii += 1

        penalty_n1_r = 0
        for jj in reversed(range(ii-5)):
            penalty_n1_r += ((jj+3) * runlength_r[jj])
            
            for kk in range(jj):
                runlength_r[kk] -= ((1+jj-kk) * runlength_r[jj])
        
        #----------------#
        #  Along Column  #
        #----------------#
        
        ii = 5
        runlength_c = list()
        while True:
        
            m_run = np.zeros((self.side_length-(ii-1), self.side_length), dtype=np.int8)
            for jj in range(ii):
                m_run += m_int8[jj:jj+self.side_length-(ii-1),:]
            
            run = int(np.sum(m_run == ii))
            if(run == 0):
                break # Completed...
            
            runlength_c.append(run)
            ii += 1

        penalty_n1_c = 0
        for jj in reversed(range(ii-5)):
            penalty_n1_c += ((jj+3) * runlength_c[jj])
            
            for kk in range(jj):
                runlength_c[kk] -= ((1+jj-kk) * runlength_c[jj])
        
        #----------------#
        #  Return Score  #
        #----------------#
    
        return (penalty_n1_r + penalty_n1_c, penalty_n2, penalty_n3_r + penalty_n3_c, penalty_n4)
    
    
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
        const_bch_poly = 0x537 # := 101_0011_0111
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
        for ii in range(4, -1, -1):
            if (i_poly>>(ii+10)) != 0 :
                i_poly ^= const_bch_poly << ii
        i_format_xor = ((i_format_info << 10) + i_poly) ^ const_qr_xor
        
        if(i_format_xor != self.format_code[i_format_info]):
            raise Exception("Format code generated ("+hex(i_format_xor)+") does not match expected ("+hex(self.format_code[i_format_info])+") for format "+hex(i_format_info)+".")
        
        #----------------------------------------------------------------------#
        #  Add format info.
        #----------------------------------------------------------------------#
        m_mask = self.cell_shaded.copy()
        
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
        
        #======================================================================#
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