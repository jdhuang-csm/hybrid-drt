import textwrap
from pathlib import Path
import numpy as np
from hybdrt.dataload.reader import read_eis, read_chrono
from hybdrt.dataload.core import FileSource


def write_file(path: Path, content: str):
    path.write_text(content)


def test_read_eis_gamry_file(tmp_path):
    # Minimal Gamry-like .dta content with CURVE TABLE and header names
    content = textwrap.dedent(
        """
        EXPLAIN
        TAG	GALVEIS
        TITLE	LABEL	Galvanostatic EIS	Test Identifier
        DATE	LABEL	03/14/2025	Date
        TIME	LABEL	16:50:03.335328	Time
        NOTES	NOTES	1	Notes...
            
        ZCURVE	TABLE
            Pt	Time	Freq	Zreal	Zimag	Zsig	Zmod	Zphz	Idc	Vdc	IERange
            #	s	Hz	ohm	ohm	V	ohm	°	A	V	#
            0	1.0	10.0	1.0	0.0	1	1.0	0.0	0.0	0.0	12
            1	2.0	1.0	2.0	-1.0	1	2.236	0.1	0.0	0.0	12
        """
    ).strip()

    p = tmp_path / "test_gamry.dta"
    write_file(p, content)

    zdata = read_eis(str(p))

    # Should return ZData-like object with expected freq and complex values
    assert np.allclose(zdata.freq, np.array([10.0, 1.0]))
    assert np.allclose(zdata.z.real, np.array([1.0, 2.0]))
    assert np.allclose(zdata.z.imag, np.array([0.0, -1.0]))


def test_read_eis_eclabtxt_file(tmp_path):
    # Minimal EC-Lab ASCII content. Use header that detect_file_source will match.
    content = textwrap.dedent(
        """
        EC-Lab ASCII FILE
        Nb header lines : 74                          

        Potentio Electrochemical Impedance Spectroscopy

        Run on channel : 6 (SN 6269)
        User : 
        Electrode connection : standard
        Potential control : Ewe
        Ewe ctrl range : min = -5.00 V, max = 5.00 V
        Ewe,I filtering : 50 kHz
        Safety Limits :
            Do not start on E overload
        Channel : Grounded
        Acquisition started on : 03/29/2024 12:21:41.631
        Loaded Setting File :  na
        Saved on :
            File : na
            Directory : na
            Host : 192.109.209.2
        Device : VMP-300 (SN 0263)
        Address : 192.109.209.31
        EC-Lab for windows v11.50 (software)
        Internet server v11.50 (firmware)
        Command interpretor v11.50 (firmware)
        Electrode material : 
        Initial state : 
        Electrolyte : 
        Comments : na
        Mass of active material : 3.180 mg
        at x = 0.000
        Molecular weight of active material (at x = 0) : 0.001 g/mol
        Atomic weight of intercalated ion : 0.001 g/mol
        Acquisition started at : xo = 0.000
        Number of e- transfered per intercalated ion : 1
        for DX = 1, DQ = 1.0 mA.h
        Battery capacity : 1.0 mA.h
        Cable : special
        Electrode surface area : 0.000 cm
        Characteristic mass : 1.0 mg
        Volume (V) : 0.001 cm
        Cycle Definition : Charge/Discharge alternance
        Mode                Single sine         
        E (V)               0.0000              
        vs.                 Emeas               
        tE (h:m:s)          0:00:0.0000         
        record              0                   
        dI                  0.000               
        unit dI             mA                  
        dt (s)              0.000               
        fi                  6.000               
        unit fi             MHz                 
        ff                  100.000             
        unit ff             mHz                 
        Nd                  10                  
        Points              per decade          
        spacing             Logarithmic         
        Va (mV)             10.0                
        pw                  0.10                
        Na                  2                   
        corr                0                   
        E range min (V)     -5.000              
        E range max (V)     5.000               
        I Range             Auto                
        Bandwidth           8                   
        nc cycles           0                   
        goto Ns'            0                   
        nr cycles           0                   
        inc. cycle          0                   

        Number of loops : 1
        Loop 0 from point number 0 to 78

        freq/Hz	Re(Z)/Ohm	-Im(Z)/Ohm	|Z|/Ohm	Phase(Z)/deg	time/s	<Ewe>/V	<I>/mA	cycle number	Ns	
        10.0	1.0	0.0	1.0	0.0	0.0	1.0E-003	1.0E-003	1.0	0
        1.0	2.0	1.0	2.236	-26.6	1.0	3.0-003	1.0E-003	1.0	0
        """
    ).lstrip()

    p = tmp_path / "test_eclab.txt"
    write_file(p, content)

    zdata = read_eis(str(p))

    assert np.allclose(zdata.freq, np.array([10.0, 1.0]))
    assert np.allclose(zdata.z.real, np.array([1.0, 2.0]))
    # eclab INVERT_Z_IM should flip sign
    assert np.allclose(zdata.z.imag, np.array([0.0, -1.0]))
    assert np.allclose(zdata.z.imag, np.array([0.0, -1.0]))


def test_read_eis_relaxis_file(tmp_path):
    content = textwrap.dedent(
        """
        RelaxIS 3.0 Spectrum export
        Date: 12/11/2025 4:05:46 PM
        Data: Frequency	Data: Z'	Data: Z''	Data: |Z|	Data: Theta (Z)
        FV2=1, AC=0.1	Model: Unassigned Spectra	(WE/RE Spectrum) test.txt		
        10.0	1.0	0.0	10629.191906554557	-0.96094740722828165
        1.0	2.0	-1.0	10627.497126875594	-1.2054814021114748
        """
    ).lstrip()

    p = tmp_path / "test_relaxis.txt"
    write_file(p, content)

    zdata = read_eis(str(p))

    assert np.allclose(zdata.freq, np.array([10.0, 1.0]))
    assert np.allclose(zdata.z.real, np.array([1.0, 2.0]))
    assert np.allclose(zdata.z.imag, np.array([0.0, -1.0]))


def test_read_chrono_eclab_file(tmp_path):
    # EC-Lab chrono file example
    content = textwrap.dedent(
        """
        EC-Lab ASCII FILE
        Nb header lines : 63                          

        Chronopotentiometry

        Run on channel : 6 (SN 12198)
        User : 
        Electrode connection : standard
        Potential control : Ewe
        Ewe ctrl range : min = 0.00 V, max = 5.00 V
        Ewe,I filtering : 50 kHz
        Safety Limits :
            Do not start on E overload
        Channel : Grounded
        Acquisition started on : 04/19/2024 11:06:01.238
        Loaded Setting File :  NONE
        Saved on :
            File : na
            Directory : na
            Host : 192.109.209.2
        Device : VMP-300 (SN 0451)
        Address : 192.109.209.30
        EC-Lab for windows v11.50 (software)
        Internet server v11.50 (firmware)
        Command interpretor v11.50 (firmware)
        Electrode material : 
        Initial state : 
        Electrolyte : 
        Comments : 240405 NCM83/LPSCl1.5 70/30
        Comments : 11.18 mg loading, GCPL, 
        Mass of active material : 7.826 mg
        at x = 0.000
        Molecular weight of active material (at x = 0) : 28.000 g/mol
        Atomic weight of intercalated ion : 7.000 g/mol
        Acquisition started at : xo = 0.000
        Number of e- transfered per intercalated ion : 1
        for DX = 1, DQ = 7.491 mA.h
        Battery capacity : 1.565 mA.h
        Cable : special
        Reference electrode : SCE Saturated Calomel Electrode (0.241 V)
        Electrode surface area : 0.001 cm²
        Characteristic mass : 7.826 mg
        Volume (V) : 0.001 cm³
        Cycle Definition : Charge/Discharge alternance
        Ns                  0                   1                   2                   3                   4                   5                   6                   7                   
        Is                  0.000               15.700              -15.700             15.700              -15.700             15.700              -15.700             0.000               
        unit Is             mA                  µA                  µA                  µA                  µA                  µA                  µA                  µA                  
        vs.                 <None>              <None>              <None>              <None>              <None>              <None>              <None>              <None>              
        ts (h:m:s)          0:00:1.0000         0:00:0.1000         0:00:0.1000         0:00:1.0000         0:00:1.0000         0:00:10.0000        0:00:10.0000        0:00:5.0000         
        EM (V)              pass                4.500               0.200               4.500               0.200               4.500               0.200               pass                
        dQM                 0.000               436.111             436.111             4.361               4.361               43.611              43.611              0.000               
        unit dQM            mA.h                pA.h                pA.h                nA.h                nA.h                nA.h                nA.h                mA.h                
        record              Ewe                 Ewe                 Ewe                 Ewe                 Ewe                 Ewe                 Ewe                 Ewe                 
        dEs (mV)            10.00               10.00               10.00               10.00               10.00               10.00               10.00               10.00               
        dts (s)             0.0010              0.0010              0.0010              0.0010              0.0010              0.0010              0.0010              0.0010              
        E range min (V)     0.000               0.000               0.000               0.000               0.000               0.000               0.000               0.000               
        E range max (V)     5.000               5.000               5.000               5.000               5.000               5.000               5.000               5.000               
        I Range             100 µA              100 µA              100 µA              100 µA              100 µA              100 µA              100 µA              100 µA              
        Bandwidth           6                   6                   6                   6                   6                   6                   6                   6                   
        goto Ns'            0                   0                   0                   0                   0                   0                   0                   0                   
        nc cycles           0                   0                   0                   0                   0                   0                   0                   0                   

        time/s	Ewe/V	I/mA	I Range				cycle number	step time/s			
        0.0	0.0	0.0	41	0	0	0	0.000000000000000E+000	0.000000000000000E+000	0	0
        1.0	0.1	1.0	41	0	0	0	0.000000000000000E+000	5.053399872340378E+000	0	0
        """
    ).strip()

    p = tmp_path / "test_chrono.txt"
    write_file(p, content)

    chrono = read_chrono(str(p))
    # TODO: need to unscale ECLAB txt files!
    assert np.allclose(chrono.time, np.array([0.0, 1.0]))
    assert np.allclose(chrono.v, np.array([0.0, 0.1]))
    assert np.allclose(chrono.i, np.array([0.0, 1e-3]))  # mA should be converted to A


