from datetime import datetime

class QR_Code_Encode :

    # def __init__(self):
    
    #-----------------#
    #  Contact VCARD  #
    #-----------------#

    def gen_str_contact(self, name, cell, ec='Q', struct_adr=None, anniversary=None, bday=None, email=None, logo=None, struct_n=None, note=None, org=None, photo=None):
    
        str_nl = '\n'
    
        str_contact = "BEGIN:VCARD" # BEGIN
        str_contact += str_nl + "VERSION:4.0" # VERSION
        str_contact += str_nl + "FN:"+str(name) # FN
        str_contact += str_nl + "TEL;TYPE=CELL:"+str(cell) # TEL
        
        #----------------------------------------------------------------------#
        def date_to_num(tuple_date):
        
            if isinstance(tuple_date, datetime):
                return str(tuple_date.year).zfill(4) + str(tuple_date.month).zfill(2) + str(tuple_date.day).zfill(2)
            
            str_month, i_date, i_year = tuple_date
            str_return = ""
            
            # Year
            str_return += str(i_year)
            
            # Month
            if type(str_month) is int :
                str_return += str(str_month).zfill(2)
            else :
                match str_month[0:3] :
                    case "Jan":
                        str_return += "01"
                    case "Feb":
                        str_return += "02"
                    case "Mar":
                        str_return += "03"
                    case "Apr":
                        str_return += "04"
                    case "May":
                        str_return += "05"
                    case "Jun":
                        str_return += "06"
                    case "Jul":
                        str_return += "07"
                    case "Aug":
                        str_return += "08"
                    case "Sep":
                        str_return += "09"
                    case "Oct":
                        str_return += "10"
                    case "Nov":
                        str_return += "11"
                    case "Dec":
                        str_return += "12"
                    
            # Date (And Return)
            str_return += str(i_date)
            return str_return
        
        #----------------------------------------------------------------------#
        
        # Stuctured...
        if struct_adr is not None :
            raise Exception("TODO")
            # po_box, aprt_suite, street, city_locality, state_region, post_code, country = struct_adr
            # str_contact += str_nl + # TODO
        
        # Date
        if anniversary is not None :
            str_contact += str_nl + "ANNIVERSARY:" + date_to_num(anniversary)
        
        # Date
        if bday is not None :
            str_contact += str_nl + "BDAY:" + date_to_num(bday)
        
        # Text
        if email is not None :
            str_contact += str_nl + "EMAIL:" + str(email)
        
        # URL <or> Base64 Encoded...
        if logo is not None :
            raise Exception("TODO")
            # str_contact += str_nl + #TODO
        
        # Stuctured...
        if struct_n is not None :
            raise Exception("TODO")
            # family_name, given_name, middle_name, honorific_pre, honorific_suf = struct_n
            # str_contact += str_nl + #TODO
        
        # Text
        if note is not None :
            str_contact += str_nl + "NOTE:" + str(note)
        
        # Structured Arbitrarily... (High --> Low) Org Unit
        if org is not None :
            raise Exception("TODO")
            # str_contact += str_nl + #TODO
        
        # URL <or> Base64 Encoded...
        if photo is not None :
            raise Exception("TODO")
            # str_contact += str_nl + #TODO
              
        # Date
        str_contact += str_nl + "REV:" + date_to_num(datetime.now()) # REV
        
        str_contact += str_nl + "END:VCARD" # END
        return str_contact
        
    #--------#
    #  WiFi  #
    #--------#
        
    def gen_str_wifi(self, auth_type, ssid, password, b_hidden=None, wpa2_eap_struct=None):
        # Special characters [\;,":] should be escaped with a backslash (\) as in MECARD encoding.
        # +-----+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        # | T   | Authentication type; can be WEP or WPA or WPA2-EAP, or nopass for no password. Or, omit for no password.
        # | S   | Network SSID. Required. Enclose in double quotes if it is an ASCII name, but could be interpreted as hex (i.e. "ABCD")
        # | P   | Password, ignored if T is nopass (in which case it may be omitted). Enclose in double quotes if it is an ASCII name, but could be interpreted as hex (i.e. "ABCD")
        # +-----+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        # | H   | Optional. True if the network SSID is hidden. 
        # |     | Note:  This was mistakenly also used to specify phase 2 method in releases up to 4.7.8 / Barcode Scanner 3.4.0. 
        # |     |        If not a boolean, it will be interpreted as phase 2 method for backwards-compatibility
        # +-----+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        # | E   | (WPA2-EAP only) EAP method, like TTLS or PWD
        # | A   | (WPA2-EAP only) Anonymous identity
        # | I   | (WPA2-EAP only) Identity
        # | PH2 | (WPA2-EAP only) Phase 2 method, like MSCHAPV2
        # +-----+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        
        t_str = "WIFI:"
        
        # TODO, IF STRING CAN BE CONSTRUED AS A HEX CODE, ADD QUOTES...
        # TODO, 5 SPECIAL CHARACTERS [\;,":]
        
        t_str += 'S:' + ssid      + ';'
        t_str += 'T:' + auth_type + ';'
        t_str += 'P:' + password  + ';'
        
        if b_hidden is not None :
            raise Exception("TODO")
            
        if wpa2_eap_struct is not None :
            e, a, i, ph2 = wpa2_eap_struct
            raise Exception("TODO")
            
        t_str += ";"
        return t_str