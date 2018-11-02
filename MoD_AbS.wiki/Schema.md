### MoD_AbS 
1.  Read parameter file
    * Directories of: continuum image, observed spectrum (_os_), output folder
    * Parameters of the disk
2.  Build continuum image
    * **build_continuum**
      * Loads continuum image, builds cube of (RMAX+something [x], RMAX+something [y], observed_spectrum at the desired resolution [z])
3.  Build cube of HI disk & determine absorption spectrum
    * **mod_abs**
      * **space**
         * builds cube with disk of size RMAX+something [x], RMAX+something [y], observed_spectrum [z], in the centre at the desired inclination (_i_) and position angle (PA), at the resolution _pix_res_
         * determines velocities along the line of sight for each pixel of the disk
         * determines which parts of the velocity disk are behind the continuum emission (_rd_)
         * the part of the disk in front of the continuum emission but are not absorbed by the continuum (_dna_, these may be due to a high cutoff of _flux_cont_lim, or because the disk is bigger of the continuum image)
         * the part of the velocity disk that is absorbing the continuum emission (_da_)
      * _da_ is used to compute the absorption spectrum: _as_ (one dimensional version of _da_)
      * _as_ is binned at the velocity resolution of the observed spectrum (_os_)
      * _rd_ and _dna_ NAN values (outside of the disk) are set to -999 for plotting
5.  Normalize model spectrum (_as_) to peak of observed absorption
* **_disk_2 == True_**
    * Redo 1-5 
    * add the two outputs for total absorption line
      * _as1_ + _as2_,  _da1_ + _da2_, _dna1_ + _dna2_, _rd1_ + _rd2_
6.  Compute residuals (model - observed spectrum) + stats
    * **chi_res**
      * Determine residual spectrum _rs_ = _as_ - _os_ within the full-width-zero intensity of _os_
      * Determine $\chi^2$ based on _rs_, noise computed in _os_ outside of the line
    * **widths**
      * Determine full-width-half-maximum (FWHM) and full-width-20%-intensity (FW20) of the model line (_as_)
7.  Write stats on table
    * **write_table**
      * write table_out in output folder with the following information
        * parameters of the model disk
        * FWHM, FW20, $\chi^2$ of the model absorption line
8.  Plot 
    * **Plot figure**
      * model observed, spectrum, residuals (_top panel_)
      * disk (_bottom panel_): absorbed (_da_, bright color), front but not absorbed (_dna_, half transparent colors), rear (_rd_, transparent colors)
        * disk is plotted in three panels
          * sky view [ra,dec] (_left panel_)
          * top view [ra, z] (_central panel_)
          * side view [dec, z] (_right panel_)
* [Read More]((asserts/MoD_AbS_readme.pdf))