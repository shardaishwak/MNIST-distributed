import re

def get_label_dic(filename):
    match = re.search(r'_(\d{2})\.png$', filename)
    if not match:
        raise ValueError(f"Filename must end with _XX.png (e.g., f0991_47.png). Got: {filename}")
    
    xx = match.group(1)
    var_name = f"LABELS_{xx}"
    print(xx)
    
    # Get from global namespace
    if var_name not in globals():
        raise ValueError(f"{var_name} is not defined. Make sure labels.py is loaded.")
    
    return globals()[var_name] # Label dictionary


# According to approx. box positions on NIST-19 forms
# First dimension is y-val, next is x-val
LABELS_57 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '73', 455: '663', 755: '9395', 1110: '85772', 1530: '996057'},
    1060 : {220: '511', 520: '0733', 875: '76567', 1290: '425260', 2000: '92'},
    1250 : {220: '8934', 575: '73692', 990: '342478', 1700: '35', 1940: '950'},
    1375 : {220: '18010', 640: '243169', 1345: '42', 1585: '258', 1880: '1088'},
    1535 : {220: '420081', 930: '16', 1170: '748', 1465: '1468', 1820: '15490'},
    1700 : {220: 'ofwacxiprdkgyjzlmvtuqbhesn'},
    1875 : {220: 'QMGEXDVYJPWFOHTCZIULBNKRSA'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_30 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '40', 455: '729', 755: '4298', 1110: '45880', 1530: '354334'},
    1060 : {220: '665', 520: '5178', 875: '45735', 1290: '236677', 2000: '76'},
    1250 : {220: '0101', 575: '65529', 990: '314887', 1700: '38', 1940: '622'},
    1375 : {220: '91265', 640: '912279', 1345: '79', 1585: '910', 1880: '8970'},
    1535 : {220: '503846', 930: '14', 1170: '036', 1465: '2048', 1820: '13109'},
    1700 : {220: 'wzxphfserlbacgjqktvnmoyiud'},
    1875 : {220: 'OBTRSYUGXDNHJZPCLIMFVEWQKA'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_75 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '23', 455: '305', 755: '6001', 1110: '85467', 1530: '723403'},
    1060 : {220: '521', 520: '0842', 875: '57996', 1290: '530367', 2000: '28'},
    1250 : {220: '4598', 575: '77182', 990: '999894', 1700: '15', 1940: '625'},
    1375 : {220: '23791', 640: '608854', 1345: '33', 1585: '191', 1880: '8604'},
    1535 : {220: '181714', 930: '32', 1170: '660', 1465: '9426', 1820: '47750'},
    1700 : {220: 'vdxtawfjgmnqcybekroiuhpslz'},
    1875 : {220: 'SANIXQBFOPDMUELWJHVGZRYCKT'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_07 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '64', 455: '264', 755: '7554', 1110: '78929', 1530: '393820'},
    1060 : {220: '980', 520: '5601', 875: '04265', 1290: '353800', 2000: '54'},
    1250 : {220: '3415', 575: '30830', 990: '627118', 1700: '17', 1940: '138'},
    1375 : {220: '54209', 640: '767416', 1345: '84', 1585: '751', 1880: '2671'},
    1535 : {220: '980694', 930: '99', 1170: '623', 1465: '7192', 1820: '25378'},
    1700 : {220: 'hgufbxeviqystcwmjdaokprnlz'},
    1875 : {220: 'BGFTPNELDJXIWHUOKRYSVCMQAZ'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_22 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '70', 455: '406', 755: '4743', 1110: '62348', 1530: '786983'},
    1060 : {220: '228', 520: '4856', 875: '50201', 1290: '129681', 2000: '21'},
    1250 : {220: '0652', 575: '97539', 990: '371838', 1700: '72', 1940: '195'},
    1375 : {220: '50119', 640: '826045', 1345: '65', 1585: '031', 1880: '8675'},
    1535 : {220: '993031', 930: '44', 1170: '049', 1465: '7326', 1820: '54797'},
    1700 : {220: 'ynbcrudqhpxvstzgmeojlifakw'},
    1875 : {220: 'HEIBCNKQMJVAWPDTYFOGURZXSL'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_68 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '45', 455: '819', 755: '3722', 1110: '35416', 1530: '917756'},
    1060 : {220: '061', 520: '6914', 875: '38183', 1290: '498387', 2000: '68'},
    1250 : {220: '2024', 575: '36007', 990: '267812', 1700: '96', 1940: '913'},
    1375 : {220: '82281', 640: '207557', 1345: '90', 1585: '306', 1880: '5139'},
    1535 : {220: '945879', 930: '07', 1170: '045', 1465: '3654', 1820: '02445'},
    1700 : {220: 'gcxoyfjrsbvmqahedtnpuwiklz'},
    1875 : {220: 'UKTJLDQNXRWYEMBVSZGIHOAPFC'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_23 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '74', 455: '040', 755: '1795', 1110: '14289', 1530: '431782'},
    1060 : {220: '443', 520: '3699', 875: '58670', 1290: '682639', 2000: '32'},
    1250 : {220: '8617', 575: '48890', 990: '339052', 1700: '50', 1940: '941'},
    1375 : {220: '03758', 640: '778297', 1345: '55', 1585: '126', 1880: '4252'},
    1535 : {220: '366500', 930: '75', 1170: '281', 1465: '6104', 1820: '31619'},
    1700 : {220: 'lhreqwupsmjtcvbozfinyxadgk'},
    1875 : {220: 'FUZDGBPLQRXKWITYJCNMOVEHSA'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_81 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '40', 455: '487', 755: '6296', 1110: '95773', 1530: '878404'},
    1060 : {220: '235', 520: '8541', 875: '86807', 1290: '672503', 2000: '05'},
    1250 : {220: '8028', 575: '74133', 990: '118091', 1700: '63', 1940: '509'},
    1375 : {220: '19575', 640: '692694', 1345: '84', 1585: '563', 1880: '4522'},
    1535 : {220: '661323', 930: '94', 1170: '220', 1465: '1997', 1820: '17301'},
    1700 : {220: 'tdcohbxurjpkyegiwszqvflnam'},
    1875 : {220: 'JGDFIOKXRNYUTPEHMLSVZWBQAC'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_00 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '09', 455: '556', 755: '5098', 1110: '98417', 1530: '735100'},
    1060 : {220: '227', 520: '8201', 875: '26337', 1290: '334666', 2000: '49'},
    1250 : {220: '1509', 575: '52820', 990: '017632', 1700: '17', 1940: '463'},
    1375 : {220: '13917', 640: '684314', 1345: '05', 1585: '369', 1880: '6175'},
    1535 : {220: '447282', 930: '25', 1170: '795', 1465: '4884', 1820: '90898'},
    1700 : {220: 'rueqdcphbziwajomgfklxtyvsn'},
    1875 : {220: 'VKGITSCEBUHLJPWDYRXMOFAQZN'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_59 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '78', 455: '712', 755: '0816', 1110: '54593', 1530: '264797'},
    1060 : {220: '014', 520: '3488', 875: '46962', 1290: '310969', 2000: '83'},
    1250 : {220: '9643', 575: '77874', 990: '538215', 1700: '45', 1940: '003'},
    1375 : {220: '29011', 640: '675592', 1345: '89', 1585: '361', 1880: '2180'},
    1535 : {220: '200387', 930: '45', 1170: '165', 1465: '2362', 1820: '49705'},
    1700 : {220: 'hoqnxvygjcwktsrelbpuiamzdf'},
    1875 : {220: 'PDJTUACKMLEVHWNGOXYSBFIRQZ'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_38 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '82', 455: '696', 755: '1999', 1110: '44381', 1530: '412509'},
    1060 : {220: '530', 520: '4917', 875: '44187', 1290: '608182', 2000: '62'},
    1250 : {220: '0912', 575: '54567', 990: '753022', 1700: '27', 1940: '503'},
    1375 : {220: '52144', 640: '300854', 1345: '79', 1585: '788', 1880: '7963'},
    1535 : {220: '179633', 930: '10', 1170: '260', 1465: '5668', 1820: '73583'},
    1700 : {220: 'yavstkqwpofceixmjlbhdzrngu'},
    1875 : {220: 'JLYRSUVOPNHCWAFXTBQMGZEIKD'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_44 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '89', 455: '261', 755: '7635', 1110: '48264', 1530: '723459'},
    1060 : {220: '203', 520: '9497', 875: '39387', 1290: '449858', 2000: '26'},
    1250 : {220: '9536', 575: '52570', 990: '231327', 1700: '31', 1940: '901'},
    1375 : {220: '70856', 640: '821350', 1345: '70', 1585: '872', 1880: '4480'},
    1535 : {220: '151460', 930: '04', 1170: '985', 1465: '1669', 1820: '07611'},
    1700 : {220: 'drfvoycmsqixnkplahjgwbtuez'},
    1875 : {220: 'ZECTGPISWXKOJLUMHVBDYNFRAQ'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_66 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '94', 455: '841', 755: '0964', 1110: '56401', 1530: '731555'},
    1060 : {220: '098', 520: '4559', 875: '17133', 1290: '581838', 2000: '63'},
    1250 : {220: '8419', 575: '62428', 990: '849537', 1700: '27', 1940: '334'},
    1375 : {220: '01260', 640: '674210', 1345: '65', 1585: '787', 1880: '5621'},
    1535 : {220: '027723', 930: '23', 1170: '296', 1465: '0890', 1820: '09976'},
    1700 : {220: 'gxieuhwntfcrkzvsmyqlopjabd'},
    1875 : {220: 'GFDBSQTPAEHNYOCRLKMUWXVZJI'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_65 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '28', 455: '013', 755: '5230', 1110: '19192', 1530: '258844'},
    1060 : {220: '976', 520: '0323', 875: '71224', 1290: '979565', 2000: '75'},
    1250 : {220: '3469', 575: '56104', 990: '859761', 1700: '73', 1940: '664'},
    1375 : {220: '17412', 640: '634338', 1345: '10', 1585: '687', 1880: '0259'},
    1535 : {220: '188907', 930: '50', 1170: '740', 1465: '8546', 1820: '32980'},
    1700 : {220: 'rdjgtxwaihvoskmquyflnbczep'},
    1875 : {220: 'LAOIJGMQEPHTCNDUKVXWZRSFYB'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_58 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '87', 455: '922', 755: '0342', 1110: '38718', 1530: '090695'},
    1060 : {220: '880', 520: '3403', 875: '57322', 1290: '645167', 2000: '19'},
    1250 : {220: '9611', 575: '26976', 990: '411778', 1700: '26', 1940: '042'},
    1375 : {220: '89340', 640: '125985', 1345: '50', 1585: '044', 1880: '5383'},
    1535 : {220: '057349', 930: '16', 1170: '251', 1465: '8549', 1820: '77636'},
    1700 : {220: 'erufvgnpxmkwdyoibjltaszqhc'},
    1875 : {220: 'LGCMYQURXSBEVFDPWJTHOANZIK'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_03 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '21', 455: '213', 755: '9985', 1110: '37077', 1530: '579947'},
    1060 : {220: '034', 520: '1447', 875: '58149', 1290: '841866', 2000: '46'},
    1250 : {220: '0553', 575: '35257', 990: '259692', 1700: '62', 1940: '120'},
    1375 : {220: '83830', 640: '874950', 1345: '97', 1585: '004', 1880: '6091'},
    1535 : {220: '627683', 930: '52', 1170: '183', 1465: '8610', 1820: '36214'},
    1700 : {220: 'ndmhqtlakbrsoejwxcygfuzivp'},
    1875 : {220: 'HQEJIBRKSYTOLDGAVNUPXWCZFM'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_16 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '22', 455: '764', 755: '0580', 1110: '73541', 1530: '805272'},
    1060 : {220: '362', 520: '1779', 875: '91246', 1290: '775450', 2000: '11'},
    1250 : {220: '9980', 575: '20356', 990: '186460', 1700: '92', 1940: '643'},
    1375 : {220: '93005', 640: '258889', 1345: '43', 1585: '428', 1880: '8557'},
    1535 : {220: '136941', 930: '38', 1170: '769', 1465: '3374', 1820: '19106'},
    1700 : {220: 'ulnmqxegpcsfdyojibhzrktawv'},
    1875 : {220: 'XGOVKMBNYTADFZJSIWCEPURHQL'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_54 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '74', 455: '509', 755: '4682', 1110: '38439', 1530: '950206'},
    1060 : {220: '995', 520: '3716', 875: '29476', 1290: '817190', 2000: '46'},
    1250 : {220: '5232', 575: '62301', 990: '633281', 1700: '46', 1940: '477'},
    1375 : {220: '12193', 640: '558379', 1345: '40', 1585: '075', 1880: '8554'},
    1535 : {220: '183170', 930: '86', 1170: '920', 1465: '6752', 1820: '80841'},
    1700 : {220: 'yorcwugjksbdmvpxnifletqhaz'},
    1875 : {220: 'BUQMXHRPDZISWAKCNVFYJOETGL'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_91 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '47', 455: '193', 755: '9467', 1110: '02704', 1530: '363893'},
    1060 : {220: '910', 520: '4441', 875: '16077', 1290: '910196', 2000: '24'},
    1250 : {220: '8656', 575: '27555', 990: '259091', 1700: '28', 1940: '720'},
    1375 : {220: '24602', 640: '652587', 1345: '28', 1585: '513', 1880: '9465'},
    1535 : {220: '387084', 930: '19', 1170: '386', 1465: '5318', 1820: '73380'},
    1700 : {220: 'rqfhuksiloxdgznvcepbwtymaj'},
    1875 : {220: 'LYJKWVMIQOFHDRNTGSUXBZPACE'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_86 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '63', 455: '565', 755: '1350', 1110: '12598', 1530: '437129'},
    1060 : {220: '980', 520: '3748', 875: '27722', 1290: '036220', 2000: '85'},
    1250 : {220: '5134', 575: '36971', 990: '469559', 1700: '88', 1940: '390'},
    1375 : {220: '71170', 640: '734982', 1345: '76', 1585: '420', 1880: '9404'},
    1535 : {220: '648301', 930: '58', 1170: '811', 1465: '5624', 1820: '67960'},
    1700 : {220: 'wgyekmburqojszpvndthfaxcil'},
    1875 : {220: 'EYORDGQASXLVTWHBKNJUIMPZFC'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_40 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '84', 455: '007', 755: '2426', 1110: '55378', 1530: '664326'},
    1060 : {220: '644', 520: '3883', 875: '01456', 1290: '478031', 2000: '90'},
    1250 : {220: '5419', 575: '12701', 990: '382927', 1700: '79', 1940: '904'},
    1375 : {220: '25865', 640: '599115', 1345: '76', 1585: '829', 1880: '1323'},
    1535 : {220: '431909', 930: '56', 1170: '368', 1465: '7010', 1820: '58277'},
    1700 : {220: 'qjkbszhxrowpiyufglvntmadec'},
    1875 : {220: 'PJBKHLVGMDWZCOSXETNFAIQURY'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_89 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '39', 455: '588', 755: '5321', 1110: '78895', 1530: '298692'},
    1060 : {220: '675', 520: '4434', 875: '75456', 1290: '239660', 2000: '17'},
    1250 : {220: '0234', 575: '08651', 990: '471361', 1700: '19', 1940: '044'},
    1375 : {220: '09577', 640: '941307', 1345: '68', 1585: '511', 1880: '1005'},
    1535 : {220: '082334', 930: '07', 1170: '722', 1465: '9632', 1820: '89862'},
    1700 : {220: 'ctmwnjysvplbuhekgqxidzrafo'},
    1875 : {220: 'PHCFJQDXMKZIAYRWBGVSOEUTNL'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_64 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '34', 455: '700', 755: '6451', 1110: '89907', 1530: '389944'},
    1060 : {220: '109', 520: '3352', 875: '72526', 1290: '282433', 2000: '64'},
    1250 : {220: '0200', 575: '91101', 990: '558667', 1700: '26', 1940: '735'},
    1375 : {220: '78498', 640: '865586', 1345: '75', 1585: '294', 1880: '2734'},
    1535 : {220: '578149', 930: '60', 1170: '681', 1465: '1302', 1820: '91317'},
    1700 : {220: 'ehdogvqbmfcpijykruslaznxtw'},
    1875 : {220: 'CPGKHSQNMBIUEOALZTXRFYDJVW'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_02 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '18', 455: '672', 755: '7114', 1110: '37789', 1530: '360364'},
    1060 : {220: '845', 520: '7560', 875: '39283', 1290: '115943', 2000: '04'},
    1250 : {220: '4096', 575: '92217', 990: '951198', 1700: '66', 1940: '900'},
    1375 : {220: '01846', 640: '882632', 1345: '92', 1585: '409', 1880: '0374'},
    1535 : {220: '583585', 930: '47', 1170: '652', 1465: '2277', 1820: '05153'},
    1700 : {220: 'lwbpjyschqnvotfzgxdekmuria'},
    1875 : {220: 'BNYXGVWEPKSDJUOLMFIHQRTCAZ'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_79 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '31', 455: '597', 755: '1706', 1110: '11302', 1530: '366244'},
    1060 : {220: '210', 520: '6331', 875: '55926', 1290: '814690', 2000: '44'},
    1250 : {220: '1859', 575: '40558', 990: '084759', 1700: '14', 1940: '293'},
    1375 : {220: '96336', 640: '563577', 1345: '77', 1585: '788', 1880: '3001'},
    1535 : {220: '890790', 930: '29', 1170: '784', 1465: '6425', 1820: '82228'},
    1700 : {220: 'cqefxmklgyjntawsirhobuvpzd'},
    1875 : {220: 'JPFASRXHMLNWBVQETYUKCGODIZ'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_62 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '96', 455: '008', 755: '3009', 1110: '60897', 1530: '357336'},
    1060 : {220: '717', 520: '1027', 875: '86148', 1290: '436451', 2000: '89'},
    1250 : {220: '4158', 575: '51403', 990: '557484', 1700: '74', 1940: '535'},
    1375 : {220: '12286', 640: '915196', 1345: '63', 1585: '028', 1880: '6930'},
    1535 : {220: '222322', 930: '64', 1170: '579', 1465: '8971', 1820: '94207'},
    1700 : {220: 'wjrfmlynqkxtbgchavpodziseu'},
    1875 : {220: 'VTRWDQLYAJUHFSIONEMPGZBKCX'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_70 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '09', 455: '835', 755: '9497', 1110: '51140', 1530: '056365'},
    1060 : {220: '015', 520: '4692', 875: '86317', 1290: '813297', 2000: '93'},
    1250 : {220: '3736', 575: '71748', 990: '749253', 1700: '43', 1940: '631'},
    1375 : {220: '02271', 640: '288681', 1345: '67', 1585: '795', 1880: '0444'},
    1535 : {220: '288590', 930: '66', 1170: '802', 1465: '4519', 1820: '25200'},
    1700 : {220: 'cijmbstrkxqdohupzeagylwvfn'},
    1875 : {220: 'XOJNTHUGWABSCQRFIVZDLPMKYE'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_80 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '13', 455: '060', 755: '7251', 1110: '50798', 1530: '941682'},
    1060 : {220: '535', 520: '3581', 875: '78798', 1290: '193954', 2000: '45'},
    1250 : {220: '7453', 575: '76109', 990: '951266', 1700: '43', 1940: '820'},
    1375 : {220: '28239', 640: '042736', 1345: '66', 1585: '472', 1880: '1729'},
    1535 : {220: '117484', 930: '60', 1170: '845', 1465: '3600', 1820: '23980'},
    1700 : {220: 'vozbqrjptxkficmudswenygahl'},
    1875 : {220: 'NOTXBDGYKZLJSMUFQRPWAEICHV'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_27 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '68', 455: '237', 755: '4686', 1110: '68017', 1530: '521735'},
    1060 : {220: '022', 520: '0254', 875: '16512', 1290: '023483', 2000: '85'},
    1250 : {220: '3659', 575: '93813', 990: '793417', 1700: '48', 1940: '396'},
    1375 : {220: '74973', 640: '096061', 1345: '05', 1585: '872', 1880: '5101'},
    1535 : {220: '094597', 930: '02', 1170: '144', 1465: '9674', 1820: '89285'},
    1700 : {220: 'rmqwzlkbdnexvcpujoifthsgay'},
    1875 : {220: 'SENALTOGQHWKVZIYMCXDPFUBJR'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_13 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '62', 455: '715', 755: '6250', 1110: '54208', 1530: '171838'},
    1060 : {220: '717', 520: '7014', 875: '44574', 1290: '497265', 2000: '39'},
    1250 : {220: '4390', 575: '87906', 990: '304637', 1700: '01', 1940: '488'},
    1375 : {220: '12931', 640: '842982', 1345: '86', 1585: '526', 1880: '9296'},
    1535 : {220: '053363', 930: '12', 1170: '503', 1465: '7615', 1820: '59809'},
    1700 : {220: 'qvpugxorkbitlfndmcyesjwhaz'},
    1875 : {220: 'SOCZGPJKXWFDIUAEVHLQYMRNBT'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_21 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '25', 455: '164', 755: '3990', 1110: '97164', 1530: '362200'},
    1060 : {220: '825', 520: '9865', 875: '70041', 1290: '743224', 2000: '13'},
    1250 : {220: '7647', 575: '77209', 990: '856438', 1700: '28', 1940: '389'},
    1375 : {220: '58054', 640: '713179', 1345: '62', 1585: '091', 1880: '7380'},
    1535 : {220: '391643', 930: '98', 1170: '218', 1465: '6450', 1820: '15565'},
    1700 : {220: 'blkgntuczdyvoisxhqmfrpjwea'},
    1875 : {220: 'SNRCYKFDVBMWPEQXGTLHIJUOZA'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_82 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '50', 455: '184', 755: '8405', 1110: '62781', 1530: '159730'},
    1060 : {220: '228', 520: '1221', 875: '75493', 1290: '362248', 2000: '10'},
    1250 : {220: '6150', 575: '80931', 990: '679391', 1700: '61', 1940: '462'},
    1375 : {220: '00298', 640: '846079', 1345: '59', 1585: '583', 1880: '5043'},
    1535 : {220: '537437', 930: '72', 1170: '359', 1465: '6947', 1820: '68467'},
    1700 : {220: 'ixkgalsuoqycptdjmrvbfewnhz'},
    1875 : {220: 'DQYHKXNFMWUSIARBLEOTCGVJZP'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_71 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '48', 455: '135', 755: '9967', 1110: '10076', 1530: '957025'},
    1060 : {220: '402', 520: '3258', 875: '89723', 1290: '795327', 2000: '11'},
    1250 : {220: '6814', 575: '62879', 990: '437413', 1700: '80', 1940: '928'},
    1375 : {220: '03611', 640: '204356', 1345: '85', 1585: '341', 1880: '8124'},
    1535 : {220: '495076', 930: '96', 1170: '587', 1465: '6506', 1820: '30294'},
    1700 : {220: 'bvdoanxgkmqstjpleifrhuywcz'},
    1875 : {220: 'JKCRYXVISNQGOTLWDEHFMPZUBA'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_37 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '12', 455: '653', 755: '0287', 1110: '03641', 1530: '436723'},
    1060 : {220: '959', 520: '1212', 875: '96013', 1290: '027576', 2000: '29'},
    1250 : {220: '1906', 575: '06020', 990: '615843', 1700: '01', 1940: '544'},
    1375 : {220: '88457', 640: '578348', 1345: '85', 1585: '297', 1880: '1381'},
    1535 : {220: '075969', 930: '47', 1170: '799', 1465: '5834', 1820: '43862'},
    1700 : {220: 'vmqgfltnacoyuekzrsjwhxipdb'},
    1875 : {220: 'STWNZYJBUKQOLICHXPREAGDMFV'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_31 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '13', 455: '594', 755: '1772', 1110: '14838', 1530: '293854'},
    1060 : {220: '889', 520: '7543', 875: '92574', 1290: '123591', 2000: '60'},
    1250 : {220: '1002', 575: '87251', 990: '185640', 1700: '47', 1940: '368'},
    1375 : {220: '03740', 640: '692658', 1345: '69', 1585: '040', 1880: '6192'},
    1535 : {220: '095137', 930: '69', 1170: '483', 1465: '7763', 1820: '05622'},
    1700 : {220: 'bijqhvwgyrfpedtoxunalckzsm'},
    1875 : {220: 'CRGNWJIABDVUHEQXLZKPYSFMOT'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_06 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '08', 455: '789', 755: '9014', 1110: '51673', 1530: '290567'},
    1060 : {220: '399', 520: '3812', 875: '96756', 1290: '350540', 2000: '34'},
    1250 : {220: '7791', 575: '43762', 990: '155050', 1700: '84', 1940: '670'},
    1375 : {220: '29344', 640: '812231', 1345: '36', 1585: '734', 1880: '9651'},
    1535 : {220: '981150', 930: '27', 1170: '866', 1465: '8284', 1820: '42280'},
    1700 : {220: 'bgcumhtpdsfklryawnjixoqvez'},
    1875 : {220: 'QLDMOTGJZWIAHFUKEPBYRSNXVC'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_42 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '76', 455: '430', 755: '6669', 1110: '00139', 1530: '457956'},
    1060 : {220: '891', 520: '3857', 875: '81485', 1290: '810761', 2000: '60'},
    1250 : {220: '3468', 575: '92174', 990: '527521', 1700: '29', 1940: '392'},
    1375 : {220: '25874', 640: '718096', 1345: '81', 1585: '157', 1880: '6354'},
    1535 : {220: '290474', 930: '32', 1170: '920', 1465: '3450', 1820: '03283'},
    1700 : {220: 'btqpscvkunorhxywladgfmijze'},
    1875 : {220: 'XDKOZSVNFCBULHIRWYQPJMETGA'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_14 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '86', 455: '506', 755: '8941', 1110: '95304', 1530: '891405'},
    1060 : {220: '521', 520: '5407', 875: '60170', 1290: '689547', 2000: '98'},
    1250 : {220: '6081', 575: '77132', 990: '314200', 1700: '78', 1940: '464'},
    1375 : {220: '93847', 640: '256369', 1345: '63', 1585: '224', 1880: '6902'},
    1535 : {220: '551339', 930: '78', 1170: '722', 1465: '5798', 1820: '21313'},
    1700 : {220: 'bgvxujdyohsmtfcwqiakrezpln'},
    1875 : {220: 'FSHKDXTEZRQMLABGVIYPUCOJWN'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_32 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '98', 455: '659', 755: '7023', 1110: '43851', 1530: '523012'},
    1060 : {220: '132', 520: '6530', 875: '72746', 1290: '405998', 2000: '95'},
    1250 : {220: '3174', 575: '76540', 990: '066206', 1700: '37', 1940: '744'},
    1375 : {220: '39289', 640: '609538', 1345: '87', 1585: '140', 1880: '4852'},
    1535 : {220: '390721', 930: '91', 1170: '517', 1465: '4861', 1820: '21688'},
    1700 : {220: 'dxijpuvsblzgnqfkwetchoarmy'},
    1875 : {220: 'MRKUHSQGCOVBEXIPDFZLYJNWAT'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_24 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '08', 455: '395', 755: '5268', 1110: '49171', 1530: '235969'},
    1060 : {220: '111', 520: '2956', 875: '81207', 1290: '758298', 2000: '90'},
    1250 : {220: '4671', 575: '34560', 990: '368704', 1700: '27', 1940: '493'},
    1375 : {220: '75434', 640: '281512', 1345: '02', 1585: '564', 1880: '3000'},
    1535 : {220: '335706', 930: '48', 1170: '863', 1465: '4699', 1820: '82771'},
    1700 : {220: 'languoybxqdtpemchrvfisjzwk'},
    1875 : {220: 'MOFWTUIXYHSDZKEBQRLPGCVNJA'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_99 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '21', 455: '063', 755: '7840', 1110: '28811', 1530: '499171'},
    1060 : {220: '816', 520: '4206', 875: '56843', 1290: '233191', 2000: '67'},
    1250 : {220: '3561', 575: '43239', 990: '026426', 1700: '68', 1940: '028'},
    1375 : {220: '14588', 640: '979765', 1345: '05', 1585: '407', 1880: '0378'},
    1535 : {220: '392545', 930: '75', 1170: '342', 1465: '5779', 1820: '90509'},
    1700 : {220: 'hfwrqsvplkibjacgznedymutxo'},
    1875 : {220: 'SKILGHXWUZDBONAVTPQMRCEFYJ'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_09 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '97', 455: '420', 755: '5290', 1110: '15880', 1530: '932784'},
    1060 : {220: '459', 520: '6104', 875: '53943', 1290: '420501', 2000: '69'},
    1250 : {220: '3291', 575: '60118', 990: '047763', 1700: '56', 1940: '607'},
    1375 : {220: '35424', 640: '183567', 1345: '52', 1585: '067', 1880: '1258'},
    1535 : {220: '193828', 930: '83', 1170: '768', 1465: '7146', 1820: '79293'},
    1700 : {220: 'ixnvlksjbuhtpwoygqefmdrcaz'},
    1875 : {220: 'EDOSMZLTUHGRXWKAFNVJYQIPCB'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_63 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '01', 455: '838', 755: '8561', 1110: '72791', 1530: '193808'},
    1060 : {220: '698', 520: '4952', 875: '11412', 1290: '808031', 2000: '16'},
    1250 : {220: '3694', 575: '57593', 990: '612702', 1700: '53', 1940: '590'},
    1375 : {220: '64495', 640: '986259', 1345: '74', 1585: '072', 1880: '8325'},
    1535 : {220: '374023', 930: '34', 1170: '674', 1465: '5027', 1820: '40766'},
    1700 : {220: 'wgtmfyipbluvjcadheqrszoknx'},
    1875 : {220: 'DLWTGBRVKMFOAHYNUPQSECZXJI'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_12 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '54', 455: '874', 755: '7739', 1110: '88315', 1530: '827426'},
    1060 : {220: '152', 520: '4558', 875: '64441', 1290: '875518', 2000: '91'},
    1250 : {220: '3633', 575: '22699', 990: '655338', 1700: '16', 1940: '568'},
    1375 : {220: '19768', 640: '374709', 1345: '00', 1585: '379', 1880: '3020'},
    1535 : {220: '101040', 930: '10', 1170: '479', 1465: '2062', 1820: '62299'},
    1700 : {220: 'rualjmcpongdxfhwstbkzviqye'},
    1875 : {220: 'LMSOZUNPTKBFHJDQEXGVIRYACW'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_72 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '09', 455: '727', 755: '4621', 1110: '98752', 1530: '321127'},
    1060 : {220: '585', 520: '8013', 875: '92381', 1290: '745243', 2000: '62'},
    1250 : {220: '5124', 575: '07154', 990: '765940', 1700: '98', 1940: '129'},
    1375 : {220: '93470', 640: '483806', 1345: '65', 1585: '695', 1880: '9850'},
    1535 : {220: '486467', 930: '81', 1170: '936', 1465: '3337', 1820: '00106'},
    1700 : {220: 'oxjlkfvuradeztynbhmpqgwics'},
    1875 : {220: 'KAJFDYBRHELITWQXMCPONZGSUV'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_36 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '07', 455: '078', 755: '3005', 1110: '53881', 1530: '849618'},
    1060 : {220: '027', 520: '3941', 875: '25989', 1290: '543033', 2000: '47'},
    1250 : {220: '9019', 575: '12791', 990: '062403', 1700: '89', 1940: '425'},
    1375 : {220: '66567', 640: '144262', 1345: '91', 1585: '555', 1880: '3148'},
    1535 : {220: '637095', 930: '67', 1170: '138', 1465: '2626', 1820: '42787'},
    1700 : {220: 'nbkmagpthrdysvufxolcjiewqz'},
    1875 : {220: 'TJMOAERYBDSIZNXVGWULFHQCKP'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_35 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '21', 455: '725', 755: '0802', 1110: '78830', 1530: '602766'},
    1060 : {220: '128', 520: '8774', 875: '77374', 1290: '543384', 2000: '54'},
    1250 : {220: '1197', 575: '43733', 990: '025566', 1700: '31', 1940: '525'},
    1375 : {220: '99841', 640: '060968', 1345: '85', 1585: '611', 1880: '9892'},
    1535 : {220: '355942', 930: '19', 1170: '491', 1465: '3920', 1820: '60406'},
    1700 : {220: 'dnfepjklovaxrhgwsbytcquzmi'},
    1875 : {220: 'VQCJGDSBKWZOLHNPXATMRFEUIY'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_43 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '83', 455: '478', 755: '6340', 1110: '97193', 1530: '847309'},
    1060 : {220: '961', 520: '4546', 875: '20621', 1290: '111724', 2000: '75'},
    1250 : {220: '2945', 575: '84297', 990: '007511', 1700: '76', 1940: '668'},
    1375 : {220: '25227', 640: '740242', 1345: '50', 1585: '333', 1880: '1896'},
    1535 : {220: '105969', 930: '85', 1170: '855', 1465: '8030', 1820: '83963'},
    1700 : {220: 'yegtpwrxlcuohmkjbnzfiqvasd'},
    1875 : {220: 'EPOKWFRJQBHDVUGNYIXTLMSACZ'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_73 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '29', 455: '053', 755: '2740', 1110: '53881', 1530: '879197'},
    1060 : {220: '595', 520: '4955', 875: '27635', 1290: '641002', 2000: '68'},
    1250 : {220: '4011', 575: '22483', 990: '247376', 1700: '38', 1940: '154'},
    1375 : {220: '98637', 640: '130921', 1345: '57', 1585: '869', 1880: '3486'},
    1535 : {220: '906714', 930: '63', 1170: '182', 1465: '9057', 1820: '24060'},
    1700 : {220: 'awcniorkebxjhgyutqfsvmldzp'},
    1875 : {220: 'IGWOPDJMSUCNFQTVKBEAYXRLHZ'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_87 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '38', 455: '632', 755: '1805', 1110: '81642', 1530: '749477'},
    1060 : {220: '809', 520: '9633', 875: '09877', 1290: '323360', 2000: '71'},
    1250 : {220: '4516', 575: '28942', 990: '853118', 1700: '46', 1940: '759'},
    1375 : {220: '21950', 640: '954713', 1345: '43', 1585: '156', 1880: '5268'},
    1535 : {220: '461627', 930: '59', 1170: '405', 1465: '9000', 1820: '02728'},
    1700 : {220: 'nyjorfghdxqvicwbtlpuaeskmz'},
    1875 : {220: 'JHOVNSCLRFUPXMGAYDTKBQEIZW'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_34 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '80', 455: '566', 755: '0802', 1110: '37947', 1530: '191714'},
    1060 : {220: '004', 520: '1757', 875: '13331', 1290: '697430', 2000: '25'},
    1250 : {220: '2608', 575: '94354', 990: '815906', 1700: '43', 1940: '633'},
    1375 : {220: '81475', 640: '722001', 1345: '77', 1585: '959', 1880: '8968'},
    1535 : {220: '823612', 930: '98', 1170: '952', 1465: '5262', 1820: '48465'},
    1700 : {220: 'yurdisoqtehxgnwfpkmcjabvlz'},
    1875 : {220: 'JFLNCURGXPYTDHZMVOKSAQWBIE'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_51 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '77', 455: '041', 755: '9294', 1110: '03639', 1530: '965472'},
    1060 : {220: '128', 520: '7102', 875: '89313', 1290: '397367', 2000: '84'},
    1250 : {220: '4234', 575: '03895', 990: '551302', 1700: '60', 1940: '291'},
    1375 : {220: '25646', 640: '618479', 1345: '15', 1585: '658', 1880: '6000'},
    1535 : {220: '455375', 930: '29', 1170: '781', 1465: '8604', 1820: '21887'},
    1700 : {220: 'bjzrcmkqhxseofaupitlvywgnd'},
    1875 : {220: 'HYQKLMENWXOFRGBUDTPVJCISAZ'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_05 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '10', 455: '456', 755: '6344', 1110: '28106', 1530: '497233'},
    1060 : {220: '920', 520: '9339', 875: '15231', 1290: '673784', 2000: '02'},
    1250 : {220: '4024', 575: '78070', 990: '693248', 1700: '60', 1940: '575'},
    1375 : {220: '10816', 640: '729795', 1345: '89', 1585: '652', 1880: '6281'},
    1535 : {220: '755735', 930: '01', 1170: '138', 1465: '4945', 1820: '18689'},
    1700 : {220: 'tjvhmqcsbgkrudwpaoxyinzelf'},
    1875 : {220: 'YUDJTSNGHKIELBPOMXVQAWCFZR'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_41 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '14', 455: '542', 755: '3309', 1110: '54308', 1530: '467077'},
    1060 : {220: '169', 520: '1293', 875: '62346', 1290: '857238', 2000: '12'},
    1250 : {220: '9588', 575: '71711', 990: '034264', 1700: '74', 1940: '274'},
    1375 : {220: '29279', 640: '286106', 1345: '85', 1585: '505', 1880: '3597'},
    1535 : {220: '485969', 930: '30', 1170: '063', 1465: '0589', 1820: '18160'},
    1700 : {220: 'zvmgticeyaskhouwdpnbxqlfjr'},
    1875 : {220: 'XZQURPCAEFBTVDOKILJYSHGWMN'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_93 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '00', 455: '584', 755: '5246', 1110: '73189', 1530: '012094'},
    1060 : {220: '857', 520: '6631', 875: '33942', 1290: '981123', 2000: '46'},
    1250 : {220: '7305', 575: '74675', 990: '168840', 1700: '56', 1940: '030'},
    1375 : {220: '07902', 640: '353211', 1345: '98', 1585: '625', 1880: '7725'},
    1535 : {220: '949548', 930: '17', 1170: '879', 1465: '2263', 1820: '86194'},
    1700 : {220: 'mzvwrncuotfxdsibghljayepqk'},
    1875 : {220: 'LQKXWCDBJFTGVOPUNYEHRIMSAZ'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_48 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '60', 455: '341', 755: '4086', 1110: '45407', 1530: '877904'},
    1060 : {220: '997', 520: '4205', 875: '85988', 1290: '407135', 2000: '31'},
    1250 : {220: '6538', 575: '73168', 990: '592209', 1700: '24', 1940: '652'},
    1375 : {220: '73136', 640: '621260', 1345: '22', 1585: '713', 1880: '7474'},
    1535 : {220: '892951', 930: '56', 1170: '835', 1465: '6800', 1820: '99311'},
    1700 : {220: 'livebncokhugstpzyxjwfdamrq'},
    1875 : {220: 'GDHKVWPLICBQYNUFXOJMRTSZEA'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_11 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '77', 455: '730', 755: '3187', 1110: '64560', 1530: '268328'},
    1060 : {220: '117', 520: '2071', 875: '60464', 1290: '458062', 2000: '31'},
    1250 : {220: '3651', 575: '85940', 990: '758837', 1700: '89', 1940: '262'},
    1375 : {220: '53173', 640: '919960', 1345: '94', 1585: '370', 1880: '4509'},
    1535 : {220: '281435', 930: '29', 1170: '402', 1465: '9165', 1820: '89425'},
    1700 : {220: 'lwpqxbdaontvrhygjusifcezkm'},
    1875 : {220: 'PWVKXLANSHCFMZRDTOUQIGYEBJ'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_61 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '80', 455: '906', 755: '1915', 1110: '73908', 1530: '952783'},
    1060 : {220: '171', 520: '2723', 875: '43383', 1290: '611295', 2000: '58'},
    1250 : {220: '0461', 575: '87766', 990: '344665', 1700: '67', 1940: '243'},
    1375 : {220: '67259', 640: '020287', 1345: '50', 1585: '219', 1880: '4432'},
    1535 : {220: '595643', 930: '54', 1170: '149', 1465: '8880', 1820: '00791'},
    1700 : {220: 'igktpnrlvcxojehsyudbwqzmfa'},
    1875 : {220: 'XLGATJUQIBSDEWZKYRHPNCOVFM'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_67 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '30', 455: '233', 755: '4815', 1110: '84968', 1530: '352137'},
    1060 : {220: '495', 520: '2122', 875: '05320', 1290: '405746', 2000: '73'},
    1250 : {220: '5774', 575: '46336', 990: '717697', 1700: '41', 1940: '880'},
    1375 : {220: '48058', 640: '962457', 1345: '51', 1585: '871', 1880: '8390'},
    1535 : {220: '582192', 930: '66', 1170: '690', 1465: '2609', 1820: '11099'},
    1700 : {220: 'fmihknotvyraqblewcjsxugdpz'},
    1875 : {220: 'PSZIKCDUFYAWBMNGXRHELTVQOJ'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_01 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '87', 455: '701', 755: '3752', 1110: '80759', 1530: '960941'},
    1060 : {220: '158', 520: '4586', 875: '32123', 1290: '832656', 2000: '82'},
    1250 : {220: '7481', 575: '80539', 990: '419219', 1700: '67', 1940: '904'},
    1375 : {220: '61738', 640: '729658', 1345: '75', 1585: '390', 1880: '5716'},
    1535 : {220: '109334', 930: '40', 1170: '625', 1465: '4234', 1820: '46002'},
    1700 : {220: 'gyxlakpdsbtzirumwfqjenhocv'},
    1875 : {220: 'ZXSBNGECMYWQTKFLUOHPIRVDJA'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_29 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '07', 455: '508', 755: '4188', 1110: '13183', 1530: '793094'},
    1060 : {220: '407', 520: '4298', 875: '72478', 1290: '931465', 2000: '22'},
    1250 : {220: '2567', 575: '87516', 990: '492935', 1700: '36', 1940: '600'},
    1375 : {220: '25649', 640: '274951', 1345: '02', 1585: '236', 1880: '1838'},
    1535 : {220: '035006', 930: '16', 1170: '953', 1465: '9458', 1820: '67117'},
    1700 : {220: 'zhbergtladjwnfkxsymipouvcq'},
    1875 : {220: 'WPZBKIJFGROMCXQLDUEASHYNVT'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_85 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '93', 455: '015', 755: '7433', 1110: '71867', 1530: '627126'},
    1060 : {220: '862', 520: '4389', 875: '70104', 1290: '990086', 2000: '30'},
    1250 : {220: '0309', 575: '73828', 990: '935502', 1700: '64', 1940: '864'},
    1375 : {220: '92458', 640: '321145', 1345: '71', 1585: '749', 1880: '6547'},
    1535 : {220: '269759', 930: '85', 1170: '618', 1465: '2214', 1820: '05531'},
    1700 : {220: 'ixfwnydlgosvphujtbecmrqkza'},
    1875 : {220: 'KDUPSXBVTNHMLFEYRGCWIQJOZA'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_49 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '05', 455: '683', 755: '9023', 1110: '33995', 1530: '644254'},
    1060 : {220: '703', 520: '9687', 875: '91526', 1290: '340416', 2000: '78'},
    1250 : {220: '4965', 575: '12667', 990: '881837', 1700: '12', 1940: '707'},
    1375 : {220: '50015', 640: '052804', 1345: '98', 1585: '318', 1880: '2995'},
    1535 : {220: '264825', 930: '41', 1170: '714', 1465: '3971', 1820: '26073'},
    1700 : {220: 'rjvgnzykqpidcmuxwsletahfbo'},
    1875 : {220: 'JGYUDSCNAIEBOQWRMHPVFTXLZK'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_46 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '52', 455: '817', 755: '2414', 1110: '14968', 1530: '453784'},
    1060 : {220: '335', 520: '6706', 875: '16870', 1290: '156085', 2000: '06'},
    1250 : {220: '1584', 575: '23976', 990: '919067', 1700: '12', 1940: '392'},
    1375 : {220: '45537', 640: '531822', 1345: '30', 1585: '294', 1880: '9700'},
    1535 : {220: '274992', 930: '81', 1170: '569', 1465: '8383', 1820: '67040'},
    1700 : {220: 'horwibevlxntkmjsfpgacyuqdz'},
    1875 : {220: 'KXSFOIWYNVRTGQDLPMBECUAHJZ'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_97 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '03', 455: '408', 755: '9921', 1110: '02238', 1530: '110837'},
    1060 : {220: '930', 520: '4861', 875: '31407', 1290: '567384', 2000: '92'},
    1250 : {220: '3800', 575: '77953', 990: '477147', 1700: '51', 1940: '265'},
    1375 : {220: '22226', 640: '151654', 1345: '90', 1585: '839', 1880: '2680'},
    1535 : {220: '564438', 930: '75', 1170: '695', 1465: '5987', 1820: '69614'},
    1700 : {220: 'rhiqpdanyuxlcosvftzkjbegwm'},
    1875 : {220: 'DPSLTEFOYRMBIUKVXHCJNQWAZG'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_60 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '96', 455: '732', 755: '4758', 1110: '60423', 1530: '756517'},
    1060 : {220: '810', 520: '0644', 875: '83429', 1290: '539695', 2000: '29'},
    1250 : {220: '9855', 575: '57660', 990: '423537', 1700: '00', 1940: '619'},
    1375 : {220: '50134', 640: '183023', 1345: '90', 1585: '811', 1880: '7429'},
    1535 : {220: '761318', 930: '08', 1170: '224', 1465: '2198', 1820: '68747'},
    1700 : {220: 'yewpinqtxvgdhouzmklbracfsj'},
    1875 : {220: 'FIBNLYTXJUGQVOEKMCWHDPRSZA'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_39 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '00', 455: '101', 755: '2753', 1110: '42440', 1530: '069665'},
    1060 : {220: '732', 520: '2344', 875: '91407', 1290: '957562', 2000: '31'},
    1250 : {220: '4409', 575: '96183', 990: '373988', 1700: '96', 1940: '888'},
    1375 : {220: '47762', 640: '198788', 1345: '72', 1585: '239', 1880: '3355'},
    1535 : {220: '079565', 930: '06', 1170: '150', 1465: '4112', 1820: '82615'},
    1700 : {220: 'lfyocqkzjmbxgduverwisnhatp'},
    1875 : {220: 'KPCANDJRMWYHFQLIBSUOEXVTZG'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_78 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '06', 455: '138', 755: '5950', 1110: '11886', 1530: '402814'},
    1060 : {220: '220', 520: '2774', 875: '61924', 1290: '614049', 2000: '76'},
    1250 : {220: '4555', 575: '69296', 990: '309038', 1700: '27', 1940: '841'},
    1375 : {220: '78673', 640: '077595', 1345: '55', 1585: '784', 1880: '3258'},
    1535 : {220: '693103', 930: '47', 1170: '118', 1465: '9332', 1820: '62903'},
    1700 : {220: 'mdyrvajinblsqgkpuoxfwechtz'},
    1875 : {220: 'LIZXBUENMPHJSVGQKRCWDYFOAT'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_52 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '19', 455: '250', 755: '2297', 1110: '20814', 1530: '798362'},
    1060 : {220: '510', 520: '6159', 875: '64113', 1290: '948078', 2000: '25'},
    1250 : {220: '5595', 575: '78460', 990: '053023', 1700: '61', 1940: '776'},
    1375 : {220: '73034', 640: '338164', 1345: '64', 1585: '141', 1880: '8792'},
    1535 : {220: '239897', 930: '96', 1170: '470', 1465: '8583', 1820: '52406'},
    1700 : {220: 'bgmjsqfvcpianwutykdelrxhoz'},
    1875 : {220: 'FXSUJNQKTLGZPHRBMAODYIWVEC'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_77 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '55', 455: '817', 755: '3942', 1110: '67059', 1530: '434535'},
    1060 : {220: '534', 520: '2739', 875: '20776', 1290: '962575', 2000: '43'},
    1250 : {220: '1646', 575: '36017', 990: '198561', 1700: '81', 1940: '930'},
    1375 : {220: '94027', 640: '833082', 1345: '60', 1585: '489', 1880: '0182'},
    1535 : {220: '612618', 930: '44', 1170: '908', 1465: '5722', 1820: '10978'},
    1700 : {220: 'ikqtaujgsfhoveyxwrcnzpmldb'},
    1875 : {220: 'NDPFCESTBAIKVLYRWJZMUXQHOG'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_33 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '81', 455: '095', 755: '7518', 1110: '69041', 1530: '938447'},
    1060 : {220: '019', 520: '2878', 875: '25960', 1290: '655333', 2000: '46'},
    1250 : {220: '9814', 575: '06100', 990: '621132', 1700: '77', 1940: '887'},
    1375 : {220: '84602', 640: '070368', 1345: '43', 1585: '715', 1880: '9937'},
    1535 : {220: '249436', 930: '22', 1170: '532', 1465: '5594', 1820: '17265'},
    1700 : {220: 'fmqeornvjtguilkcywdxphbasz'},
    1875 : {220: 'JBYNHUFQOGIZWTCLPEVXRASMKD'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_15 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '35', 455: '329', 755: '3214', 1110: '55232', 1530: '139721'},
    1060 : {220: '289', 520: '1887', 875: '81006', 1290: '778750', 2000: '61'},
    1250 : {220: '5746', 575: '12507', 990: '990384', 1700: '48', 1940: '418'},
    1375 : {220: '65900', 640: '037164', 1345: '26', 1585: '604', 1880: '5413'},
    1535 : {220: '863995', 930: '93', 1170: '785', 1465: '6476', 1820: '22094'},
    1700 : {220: 'rgnvdpujhkyibmoaefscltwxqz'},
    1875 : {220: 'HNLQOCSXVTEYWZJMUARGFKIDBP'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_56 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '71', 455: '704', 755: '5766', 1110: '53925', 1530: '815384'},
    1060 : {220: '884', 520: '5128', 875: '50260', 1290: '266626', 2000: '85'},
    1250 : {220: '3197', 575: '00573', 990: '900157', 1700: '49', 1940: '173'},
    1375 : {220: '01398', 640: '444436', 1345: '52', 1585: '983', 1880: '2811'},
    1535 : {220: '136790', 930: '49', 1170: '462', 1465: '9907', 1820: '27328'},
    1700 : {220: 'efmjlqbgzntwaiysphkxdruvoc'},
    1875 : {220: 'YIQNMDSEJLGXZFOHUCKTRWVBPA'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_84 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '09', 455: '843', 755: '5466', 1110: '76625', 1530: '241122'},
    1060 : {220: '258', 520: '8107', 875: '29160', 1290: '757313', 2000: '98'},
    1250 : {220: '1862', 575: '78648', 990: '451094', 1700: '60', 1940: '816'},
    1375 : {220: '87345', 640: '457792', 1345: '29', 1585: '377', 1880: '9538'},
    1535 : {220: '042503', 930: '03', 1170: '915', 1465: '1933', 1820: '94600'},
    1700 : {220: 'caqokhvzmdelntfgbijuxsywpr'},
    1875 : {220: 'EYGXJOSLWNQBMTUDHKFAPZVIRC'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_45 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '34', 455: '146', 755: '6733', 1110: '21947', 1530: '354911'},
    1060 : {220: '014', 520: '6028', 875: '25217', 1290: '500206', 2000: '82'},
    1250 : {220: '9237', 575: '69878', 990: '922986', 1700: '95', 1940: '444'},
    1375 : {220: '00808', 640: '668517', 1345: '54', 1585: '385', 1880: '0367'},
    1535 : {220: '831395', 930: '15', 1170: '995', 1465: '0727', 1820: '43176'},
    1700 : {220: 'zhxydqunlogmiwbpsfvctkjrae'},
    1875 : {220: 'KBDMCOWQITLHPUXFRSVEYNJGZA'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_95 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '15', 455: '082', 755: '7651', 1110: '40324', 1530: '325998'},
    1060 : {220: '167', 520: '7004', 875: '78741', 1290: '998946', 2000: '16'},
    1250 : {220: '1883', 575: '18091', 990: '930470', 1700: '03', 1940: '581'},
    1375 : {220: '27936', 640: '527517', 1345: '75', 1585: '696', 1880: '2902'},
    1535 : {220: '434035', 930: '82', 1170: '482', 1465: '6323', 1820: '46556'},
    1700 : {220: 'psdrvejzbytlogimnukxwqhfac'},
    1875 : {220: 'MCDIEZSOAHTNQVKWFBLGJYPRXU'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_10 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '16', 455: '899', 755: '0124', 1110: '43744', 1530: '403875'},
    1060 : {220: '821', 520: '7358', 875: '55925', 1290: '116213', 2000: '86'},
    1250 : {220: '4949', 575: '20044', 990: '625502', 1700: '80', 1940: '681'},
    1375 : {220: '79192', 640: '676687', 1345: '49', 1585: '213', 1880: '3055'},
    1535 : {220: '803797', 930: '02', 1170: '791', 1465: '3678', 1820: '03536'},
    1700 : {220: 'denayxlghmkufwvqsoitrjcbpz'},
    1875 : {220: 'FTUJSPNOIYMQELHDXGWAKVBRZC'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_90 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '24', 455: '850', 755: '3756', 1110: '00468', 1530: '625578'},
    1060 : {220: '107', 520: '8890', 875: '78269', 1290: '444295', 2000: '33'},
    1250 : {220: '5621', 575: '80753', 990: '352773', 1700: '63', 1940: '473'},
    1375 : {220: '21214', 640: '131248', 1345: '97', 1585: '429', 1880: '8991'},
    1535 : {220: '968150', 930: '94', 1170: '666', 1465: '1130', 1820: '50097'},
    1700 : {220: 'mowxyjerhflnvcubgsdkiqpzat'},
    1875 : {220: 'YVBKIMJCHOLNFZDGERXPAWTUQS'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_76 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '02', 455: '324', 755: '8101', 1110: '36752', 1530: '971666'},
    1060 : {220: '889', 520: '9348', 875: '95557', 1290: '828710', 2000: '57'},
    1250 : {220: '0001', 575: '78874', 990: '677129', 1700: '15', 1940: '244'},
    1375 : {220: '89933', 640: '564692', 1345: '61', 1585: '341', 1880: '5303'},
    1535 : {220: '205864', 930: '57', 1170: '424', 1465: '0291', 1820: '06393'},
    1700 : {220: 'vhdljnwyotbcirfzugsxqkmape'},
    1875 : {220: 'COBJGHWNUYIFETAXLRSDKQPMVZ'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_98 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '02', 455: '559', 755: '5979', 1110: '62812', 1530: '261835'},
    1060 : {220: '444', 520: '5817', 875: '47875', 1290: '717541', 2000: '93'},
    1250 : {220: '7806', 575: '30412', 990: '674648', 1700: '04', 1940: '499'},
    1375 : {220: '02391', 640: '633600', 1345: '63', 1585: '282', 1880: '8213'},
    1535 : {220: '513989', 930: '55', 1170: '876', 1465: '0307', 1820: '29061'},
    1700 : {220: 'ateyxlbuogscmjdiwrqpfhnvzk'},
    1875 : {220: 'LYMDOTZBKUXWQEGARHINVJPFSC'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_53 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '83', 455: '310', 755: '4191', 1110: '18789', 1530: '464301'},
    1060 : {220: '814', 520: '9046', 875: '78523', 1290: '092351', 2000: '07'},
    1250 : {220: '9522', 575: '09260', 990: '669062', 1700: '25', 1940: '277'},
    1375 : {220: '13244', 640: '163828', 1345: '79', 1585: '570', 1880: '4375'},
    1535 : {220: '389947', 930: '55', 1170: '635', 1465: '7064', 1820: '56818'},
    1700 : {220: 'ckdlzverfyqojiuhwbnmstaxpg'},
    1875 : {220: 'OJEAMLPZVBKWNDYRFSHUCIXTGQ'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_96 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '49', 455: '068', 755: '3274', 1110: '32501', 1530: '604257'},
    1060 : {220: '260', 520: '8612', 875: '09460', 1290: '582796', 2000: '67'},
    1250 : {220: '3803', 575: '36146', 990: '853197', 1700: '38', 1940: '891'},
    1375 : {220: '73157', 640: '494921', 1345: '22', 1585: '814', 1880: '8072'},
    1535 : {220: '054491', 930: '63', 1170: '553', 1465: '1587', 1820: '95970'},
    1700 : {220: 'lsmzvcfxphuqnatyrwjdkogebi'},
    1875 : {220: 'WANXMTLRVQGEPKSUIOHYDFBCJZ'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_17 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '70', 455: '607', 755: '3867', 1110: '13797', 1530: '402506'},
    1060 : {220: '495', 520: '6529', 875: '85153', 1290: '983405', 2000: '48'},
    1250 : {220: '6348', 575: '06481', 990: '340192', 1700: '37', 1940: '912'},
    1375 : {220: '51728', 640: '701964', 1345: '92', 1585: '123', 1880: '7021'},
    1535 : {220: '353472', 930: '58', 1170: '168', 1465: '5608', 1820: '94926'},
    1700 : {220: 'ylhpgxqzanrvjisomuebdtckfw'},
    1875 : {220: 'STNXVAOCHGDQIFWUPKMJLREBZY'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_94 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '36', 455: '198', 755: '2348', 1110: '51983', 1530: '629861'},
    1060 : {220: '521', 520: '3457', 875: '89630', 1290: '770458', 2000: '23'},
    1250 : {220: '4748', 575: '31055', 990: '863279', 1700: '51', 1940: '612'},
    1375 : {220: '24809', 640: '256821', 1345: '60', 1585: '759', 1880: '7910'},
    1535 : {220: '300046', 930: '36', 1170: '470', 1465: '7795', 1820: '41429'},
    1700 : {220: 'gybspejuxvtwlkiqorcmfanhzd'},
    1875 : {220: 'ECYIPRHBGXONJLUVTKWDSQMFAZ'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_28 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '09', 455: '404', 755: '2415', 1110: '61178', 1530: '534670'},
    1060 : {220: '330', 520: '2698', 875: '58689', 1290: '977211', 2000: '12'},
    1250 : {220: '8057', 575: '65381', 990: '360489', 1700: '49', 1940: '795'},
    1375 : {220: '27182', 640: '177320', 1345: '46', 1585: '685', 1880: '9428'},
    1535 : {220: '251050', 930: '34', 1170: '396', 1465: '2573', 1820: '43069'},
    1700 : {220: 'nbcqvujxgdkiwrfhpyetzaolsm'},
    1875 : {220: 'THKSWOVGAPDJUMNYQCLIXBFZER'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_20 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '74', 455: '385', 755: '1852', 1110: '86192', 1530: '409937'},
    1060 : {220: '791', 520: '8442', 875: '75853', 1290: '220586', 2000: '04'},
    1250 : {220: '0381', 575: '03047', 990: '492912', 1700: '57', 1940: '171'},
    1375 : {220: '67336', 640: '562876', 1345: '49', 1585: '882', 1880: '9655'},
    1535 : {220: '953743', 930: '60', 1170: '046', 1465: '6901', 1820: '13210'},
    1700 : {220: 'vgutednlscmwrofjxhpkyqbiza'},
    1875 : {220: 'XGRJCWODLMTPYQAFKHUENBISVZ'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_55 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '79', 455: '001', 755: '4196', 1110: '49856', 1530: '866305'},
    1060 : {220: '187', 520: '9559', 875: '22321', 1290: '077925', 2000: '69'},
    1250 : {220: '8131', 575: '44306', 990: '420243', 1700: '38', 1940: '628'},
    1375 : {220: '59501', 640: '778320', 1345: '81', 1585: '544', 1880: '4977'},
    1535 : {220: '251364', 930: '57', 1170: '138', 1465: '6708', 1820: '26390'},
    1700 : {220: 'cmdvwbpjyrhqkaleztougfsnxi'},
    1875 : {220: 'JDAWTFPNGHOSBQRULYKVCIMEZX'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_47 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '42', 455: '021', 755: '5942', 1110: '65935', 1530: '071153'},
    1060 : {220: '079', 520: '6167', 875: '49579', 1290: '856248', 2000: '15'},
    1250 : {220: '9228', 575: '06433', 990: '124158', 1700: '96', 1940: '934'},
    1375 : {220: '03902', 640: '711076', 1345: '22', 1585: '760', 1880: '6948'},
    1535 : {220: '433781', 930: '04', 1170: '583', 1465: '0683', 1820: '78578'},
    1700 : {220: 'cyogkmnijpqtevufslbarwzdxh'},
    1875 : {220: 'ZFLHEXYPJTVSBUMRGDONIQKCWA'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_08 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '75', 455: '519', 755: '9710', 1110: '90597', 1530: '172236'},
    1060 : {220: '808', 520: '3209', 875: '06175', 1290: '834766', 2000: '32'},
    1250 : {220: '3894', 575: '88710', 990: '877592', 1700: '85', 1940: '346'},
    1375 : {220: '11455', 640: '048749', 1345: '23', 1585: '641', 1880: '2415'},
    1535 : {220: '420486', 930: '19', 1170: '320', 1465: '0256', 1820: '93636'},
    1700 : {220: 'garvnmwixhsdypjqofekzultcb'},
    1875 : {220: 'OLTZPCUVAFQHIDJMRNGKSBWXYE'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_83 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '50', 455: '267', 755: '3054', 1110: '34309', 1530: '072886'},
    1060 : {220: '701', 520: '1818', 875: '79541', 1290: '692693', 2000: '08'},
    1250 : {220: '6508', 575: '74411', 990: '903126', 1700: '26', 1940: '787'},
    1375 : {220: '29783', 640: '411026', 1345: '27', 1585: '355', 1880: '3584'},
    1535 : {220: '275491', 930: '99', 1170: '945', 1465: '8462', 1820: '53603'},
    1700 : {220: 'vlirpbwcdqmstoueyfngzahkjx'},
    1875 : {220: 'AHVUPXLDICYFNTSGJWZEQMKROB'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_04 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '17', 455: '481', 755: '5657', 1110: '28633', 1530: '865409'},
    1060 : {220: '172', 520: '9151', 875: '32230', 1290: '643769', 2000: '04'},
    1250 : {220: '8140', 575: '61269', 990: '223551', 1700: '07', 1940: '796'},
    1375 : {220: '29470', 640: '234008', 1345: '88', 1585: '513', 1880: '7498'},
    1535 : {220: '890989', 930: '02', 1170: '656', 1465: '7475', 1820: '41353'},
    1700 : {220: 'whxlsbtyfodijurvpqcenmkzag'},
    1875 : {220: 'GWHYRBEDFJILUPZOKXQSMCTNVA'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_92 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '80', 455: '433', 755: '9977', 1110: '02421', 1530: '127939'},
    1060 : {220: '461', 520: '3234', 875: '11587', 1290: '187514', 2000: '43'},
    1250 : {220: '6170', 575: '57506', 990: '905921', 1700: '60', 1940: '845'},
    1375 : {220: '43156', 640: '085673', 1345: '27', 1585: '200', 1880: '9648'},
    1535 : {220: '956329', 930: '65', 1170: '029', 1465: '8486', 1820: '83872'},
    1700 : {220: 'dntmgbfiexrujwkophvalcqyzs'},
    1875 : {220: 'PFGLQEYAJUNRDMOXHSTBWKVICZ'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_69 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '61', 455: '095', 755: '5547', 1110: '17933', 1530: '861759'},
    1060 : {220: '198', 520: '9777', 875: '24263', 1290: '205687', 2000: '58'},
    1250 : {220: '3903', 575: '94841', 990: '826808', 1700: '82', 1940: '701'},
    1375 : {220: '23239', 640: '554119', 1345: '25', 1585: '536', 1880: '3742'},
    1535 : {220: '444000', 930: '86', 1170: '116', 1465: '2034', 1820: '76069'},
    1700 : {220: 'feocmpilubraswdyhjnxgtvqzk'},
    1875 : {220: 'WTDSUYEAOZHBRLKVQJCMFIXNPG'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_18 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '89', 455: '570', 755: '3168', 1110: '41765', 1530: '642781'},
    1060 : {220: '343', 520: '4720', 875: '50192', 1290: '323557', 2000: '84'},
    1250 : {220: '9971', 575: '19078', 990: '348638', 1700: '09', 1940: '621'},
    1375 : {220: '01062', 640: '389072', 1345: '34', 1585: '552', 1880: '8546'},
    1535 : {220: '667918', 930: '21', 1170: '265', 1465: '3479', 1820: '40059'},
    1700 : {220: 'dypgvckrsjnlfituwqxohbemaz'},
    1875 : {220: 'SATWKLRDPMBVUNYXIJFGOQEHCZ'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_25 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '38', 455: '071', 755: '0755', 1110: '69010', 1530: '083431'},
    1060 : {220: '500', 520: '9534', 875: '93769', 1290: '245721', 2000: '64'},
    1250 : {220: '9494', 575: '12258', 990: '132943', 1700: '82', 1940: '822'},
    1375 : {220: '12865', 640: '167213', 1345: '75', 1585: '938', 1880: '7570'},
    1535 : {220: '748850', 930: '66', 1170: '376', 1465: '9948', 1820: '41066'},
    1700 : {220: 'brwnjhlqysopeictkvzufdmxga'},
    1875 : {220: 'UJSKHCLMXPEOWFVNRDGYZIQBTA'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_74 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '77', 455: '218', 755: '6802', 1110: '75428', 1530: '636973'},
    1060 : {220: '706', 520: '1148', 875: '28011', 1290: '656421', 2000: '93'},
    1250 : {220: '0995', 575: '61555', 990: '297440', 1700: '23', 1940: '249'},
    1375 : {220: '10035', 640: '381675', 1345: '84', 1585: '686', 1880: '5909'},
    1535 : {220: '251430', 930: '07', 1170: '938', 1465: '2487', 1820: '94337'},
    1700 : {220: 'vhjlezxmtbkgpuswcifyqrnoda'},
    1875 : {220: 'WTLIJHOEGMQBPRXASUNFVDCZYK'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_88 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '91', 455: '707', 755: '3495', 1110: '08184', 1530: '652687'},
    1060 : {220: '503', 520: '9236', 875: '51094', 1290: '278738', 2000: '14'},
    1250 : {220: '6473', 575: '16974', 990: '139465', 1700: '22', 1940: '822'},
    1375 : {220: '90871', 640: '610163', 1345: '29', 1585: '807', 1880: '4685'},
    1535 : {220: '355552', 930: '40', 1170: '603', 1465: '8370', 1820: '21949'},
    1700 : {220: 'bhvxowgajdcqykfntspeziumrl'},
    1875 : {220: 'DCEWTRMQJYSZFIBOUGPKHLVANX'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_19 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '85', 455: '795', 755: '0835', 1110: '48302', 1530: '102588'},
    1060 : {220: '530', 520: '4416', 875: '13036', 1290: '937497', 2000: '09'},
    1250 : {220: '4832', 575: '77957', 990: '652489', 1700: '69', 1940: '764'},
    1375 : {220: '12061', 640: '172364', 1345: '89', 1585: '646', 1880: '5021'},
    1535 : {220: '056501', 930: '21', 1170: '781', 1465: '4328', 1820: '27939'},
    1700 : {220: 'enwygluchjvsxtmqfrpadoibkz'},
    1875 : {220: 'VDRHBXSELYGZKANTFJQWUMOPCI'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_50 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '17', 455: '298', 755: '8118', 1110: '84638', 1530: '211763'},
    1060 : {220: '666', 520: '3724', 875: '26950', 1290: '294277', 2000: '45'},
    1250 : {220: '7509', 575: '39024', 990: '455538', 1700: '46', 1940: '729'},
    1375 : {220: '99536', 640: '950853', 1345: '83', 1585: '661', 1880: '4789'},
    1535 : {220: '037402', 930: '21', 1170: '011', 1465: '7430', 1820: '15800'},
    1700 : {220: 'twbsrkvmoixupfqezyngcldjha'},
    1875 : {220: 'PYMIRDNSKFEJLXCWHUGTAVQOZB'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

LABELS_26 = {
    740 : {220: '0123456789', 870: '0123456789', 1530: '0123456789'},
    900 : {220: '57', 455: '770', 755: '4031', 1110: '89218', 1530: '654734'},
    1060 : {220: '186', 520: '9145', 875: '56013', 1290: '648985', 2000: '21'},
    1250 : {220: '0725', 575: '56663', 990: '331722', 1700: '72', 1940: '368'},
    1375 : {220: '51490', 640: '506880', 1345: '82', 1585: '394', 1880: '3717'},
    1535 : {220: '694299', 930: '20', 1170: '850', 1465: '4940', 1820: '71329'},
    1700 : {220: 'vphwluzqydbgaxekjimtconfrs'},
    1875 : {220: 'UTBQORIJSLXYDHFAGNKPWEVCMZ'},
    2230 : {220: 'We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,establishJustice,insuredomesticTranquility,provideforthecommonDefense,promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica.'},
}

