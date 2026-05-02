set dumpfile waveform.vcd
gtkwave::loadFile $dumpfile

set all_sigs [gtkwave::getDisplayedSignals]

# Add signals of interest
set sigs {
    tb_waveform.clk
    tb_waveform.cs_n
    tb_waveform.sck
    tb_waveform.mosi
    tb_waveform.miso
    tb_waveform.spike_in
    tb_waveform.start
    tb_waveform.weight_wr_en
}

foreach s $sigs {
    gtkwave::addSignalsFromList $s
}

gtkwave::setZoomFactor -30
gtkwave::setFromEntry 0
gtkwave::setToEntry 12000

gtkwave::hardcopy ../sim/waveform.png png

gtkwave::quit
