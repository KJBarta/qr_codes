import numpy as np
import qr_code

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

#------------------------------------------------------------------------------#
# FNC1 Mode:
#   - Messages containing specific data formats.
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# Default Interpretation for QR Code == ECI 000003 == ISO/IEC 8859-1 Character Set
#   - Use ECI protocol for other character sets
#   - Extended Channel Interpretations
#------------------------------------------------------------------------------#

class QR_Code_Gen :
    
    #------------------#
    #  Initialization  #
    #------------------#
    
    def __init__(self):
    
        # #unused bits.
        self.v_to_remainder = [None, 0, 7,7,7,7,7, 0,0,0,0,0,0,0, 3,3,3,3,3,3,3, 4,4,4,4,4,4,4, 3,3,3,3,3,3,3, 0,0,0,0,0,0]
        
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
    
    def get_string_table_info(self):
        t_str = ""
        for ii in range(1,41):
            t_str += '['+str(ii)+']  '+str(self.v_to_ec_codewords[ii])+'  =  '+str(self.v_to_blocks[ii])+'  .*  '+str(self.v_to_ec_per_block[ii])
            for rs_level in ['L','M','Q','H'] :
                if  self.v_to_ec_codewords[ii][rs_level] !=  (self.v_to_blocks[ii][rs_level] * self.v_to_ec_per_block[ii][rs_level]) :
                    t_str += "\t["+str(ii)+"]["+rs_level+"]  ::  "+str(self.v_to_ec_codewords[ii][rs_level])+"  !=  "+str(self.v_to_blocks[ii][rs_level])+"  *  "+str(self.v_to_ec_per_block[ii][rs_level])
        return t_str
    
    def get_string_help(self):
        #------------------------------------------------------------------------------#
        # Data Capacity @ v40 L:
        #   - Numeric: 7089
        #   - Alpha:   4296
        #   - Byte:    2953
        #------------------------------------------------------------------------------#
        return ""
    
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
                    return 10 + i_char - ord('A')
                case 0x20: # ord(' '): # x20
                    return 36
                case 0x23: # ord('#'): # x23
                    return 37
                case 0x25: # ord('%'): # x25
                    return 38
                case 0x2A: # ord('*'): # x2A
                    return 39
                case 0x2B: # ord('+'): # x2B
                    return 40
                case 0x2D: # ord('-'): # x2D
                    return 41
                case 0x2E: # ord('.'): # x2E
                    return 42
                case 0x2F: # ord('/'): # x2F
                    return 43
                case 0x3A: # ord(':'): # x3A
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
        if False:
            raise Exception("<>.")
        
        #----------------------------------------------------------------------#
        # Parse Numeric Data
        #----------------------------------------------------------------------#
        (i_data, i_count) = self.parse_numeric_encoding(in_str)
        
        #----------------------------------------------------------------------#
        # Numeric Mode, Char Count Bits:  [V1-V9]=10 [V10-V26]=12 [V27-V40]=14
        # (Segment + Terminator) = (Mode + Char Count + Data + Terminator)
        #----------------------------------------------------------------------#
        for ii in range(1,41):
            
            #------------------------------------------------------------------#
            if ii <= 10 :
                i_bits = 10
            elif ii <= 27 :
                i_bits = 12
            else :
                i_bits = 14
            
            #------------------------------------------------------------------#
            if (self.v_to_num_codewords[ii] - self.v_to_ec_codewords[ii][ec]) >= int( np.ceil((4 + i_bits + i_count + 4)/8) ) :
                           
                # Construct segment.
                t_mode = 1
                t_segment = ((((t_mode<<i_bits) + len(in_str))<<i_count) + i_data)<<4
                t_len = 4 + i_bits + i_count + 4
                
                # Pad with 0s if not a multiple of 8.
                t_segment <<= (8 - (t_len % 8))
                
                # Convert to numpy-array.
                segment_matrix = np.zeros( np.ceil(t_len/8).astype(int), dtype=np.uint8)
                for jj in range( np.ceil(t_len/8).astype(int)-1, -1, -1):
                    segment_matrix[jj] = (t_segment >> (jj<<3)) % (1<<8)
                segment_matrix = np.flip(segment_matrix)
                                
                # Return QR Code
                return qr_code.QR_Code(                                       \
                    i_qr_ver        = ii,                             \
                    c_ec            = ec,                             \
                    m_data          = segment_matrix,                 \
                    i_num_codewords = self.v_to_num_codewords[ii],    \
                    i_num_blocks    = self.v_to_blocks[ii][ec],       \
                    i_ec_per_block  = self.v_to_ec_per_block[ii][ec], \
                    l_alignment     = self.v_to_alignment[ii]         \
                )
        
        raise Exception("Number too large to be represented by QR code V40.")
        
    def gen_QR_alphanumeric_encoding(self, in_str, ec='Q'): # 0-9 A-Z $ % * + - . / : (Space)
    
        #----------------------------------------------------------------------#
        # TODO: Regular Expression check...
        #----------------------------------------------------------------------#
        if False:
            raise Exception("<>.")
        
        #----------------------------------------------------------------------#
        # Parse Numeric Data
        #----------------------------------------------------------------------#
        (i_data, i_count) = self.parse_alphanumeric_encoding(in_str)
        
        #----------------------------------------------------------------------#
        # Numeric Mode, Char Count Bits:  [V1-V9]=10 [V10-V26]=12 [V27-V40]=14
        # (Segment + Terminator) = (Mode + Char Count + Data + Terminator)
        #----------------------------------------------------------------------#
        for ii in range(1,41): # 9 bits in char count.
            
            #------------------------------------------------------------------#
            if ii <= 10 :
                i_bits = 9
            elif ii <= 27 :
                i_bits = 11
            else :
                i_bits = 13
            
            #------------------------------------------------------------------#
            if (self.v_to_num_codewords[ii] - self.v_to_ec_codewords[ii][ec]) >= int( np.ceil((4 + i_bits + i_count + 4)/8) ) :
                           
                # Construct segment.
                t_mode = 2
                t_segment = ((((t_mode<<i_bits) + len(in_str))<<i_count) + i_data)<<4
                t_len = 4 + i_bits + i_count + 4
                
                # Pad with 0s if not a multiple of 8.
                t_segment <<= (8 - (t_len % 8))
                
                # Convert to numpy-array.
                segment_matrix = np.zeros( np.ceil(t_len/8).astype(int), dtype=np.uint8)
                for jj in range( np.ceil(t_len/8).astype(int)-1, -1, -1):
                    segment_matrix[jj] = (t_segment >> (jj<<3)) % (1<<8)
                segment_matrix = np.flip(segment_matrix)
                                
                # Return QR Code
                return qr_code.QR_Code(                                       \
                    i_qr_ver        = ii,                             \
                    c_ec            = ec,                             \
                    m_data          = segment_matrix,                 \
                    i_num_codewords = self.v_to_num_codewords[ii],    \
                    i_num_blocks    = self.v_to_blocks[ii][ec],       \
                    i_ec_per_block  = self.v_to_ec_per_block[ii][ec], \
                    l_alignment     = self.v_to_alignment[ii]         \
                )
        
        raise Exception("Alphanumeric too large to be represented by QR code V40.")
    
    def gen_QR_byte_encoding(self, in_str, ec='Q'): # 8-bits per character.
    
        #----------------------------------------------------------------------#
        # TODO: Regular Expression check...
        #----------------------------------------------------------------------#
        if False:
            raise Exception("<>.")
        
        #----------------------------------------------------------------------#
        # Parse Numeric Data
        #----------------------------------------------------------------------#
        (i_data, i_count) = self.parse_byte_encoding(in_str)
        
        #----------------------------------------------------------------------#
        # Numeric Mode, Char Count Bits:  [V1-V9]=10 [V10-V26]=12 [V27-V40]=14
        # (Segment + Terminator) = (Mode + Char Count + Data + Terminator)
        #----------------------------------------------------------------------#
        for ii in range(1,41): # 8 bits in char count.
        
            #------------------------------------------------------------------#
            if ii <= 10 :
                i_bits = 8
            elif ii <= 27 :
                i_bits = 16
            else :
                i_bits = 16
            
            #------------------------------------------------------------------#
            if (self.v_to_num_codewords[ii] - self.v_to_ec_codewords[ii][ec]) >= int( np.ceil((4 + i_bits + i_count + 4)/8) ) :
                           
                # Construct segment.
                t_mode = 4
                t_segment = ((((t_mode<<i_bits) + len(in_str))<<i_count) + i_data)<<4
                t_len = 4 + i_bits + i_count + 4
                
                # Pad with 0s if not a multiple of 8.
                t_segment <<= (8 - t_len)%8
                
                # Convert to numpy-array.
                segment_matrix = np.zeros( np.ceil(t_len/8).astype(int), dtype=np.uint8)
                for jj in range( np.ceil(t_len/8).astype(int)-1, -1, -1):
                    segment_matrix[jj] = (t_segment >> (jj<<3)) % (1<<8)
                segment_matrix = np.flip(segment_matrix)
                                
                # Return QR Code
                return qr_code.QR_Code(                                       \
                    i_qr_ver        = ii,                             \
                    c_ec            = ec,                             \
                    m_data          = segment_matrix,                 \
                    i_num_codewords = self.v_to_num_codewords[ii],    \
                    i_num_blocks    = self.v_to_blocks[ii][ec],       \
                    i_ec_per_block  = self.v_to_ec_per_block[ii][ec], \
                    l_alignment     = self.v_to_alignment[ii]         \
                )
        
        raise Exception("Byte too large to be represented by QR code V40.")
        
    # def gen_QR_kanji_encoding(self, in_str):
        # return qr_code.QR_Code()
        
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
    # def gen_QR(self, in_str):
    
        # # Error if not string.
        # if not isinstance(in_str, str):
            # raise Exception("Expecting input type 'str'.")
        
        # # Decode data:
        # data = None
        
        # # Determine how big QR needed based on data.
        # necessary_codewords = None
        # ver = 1
        # while( necessary_codewords > self.v_to_num_codewords[ii] ):
            # ver += 1
            # if ver > 40 :
                # raise Exception("Byte data exceeds QR Code v40 size.")
        
        # # Generate QR Code class.
        # return qr_code.QR_Code(ver, data)