titratable_residues = ['PRN', 'PAT']

# Propionate (amber's definition)
refene1 = _ReferenceEnergy(igb1=0, igb2=0, igb5=0, igb7=0, igb8=0)
refene1.solvent_energies(igb1=0, igb2=0, igb5=0, igb7=0, igb8=0)
refene1.dielc2_energies(igb1=0, igb2=0, igb5=0, igb7=0, igb8=0)
refene1.dielc2.solvent_energies(igb1=0, igb2=0, igb5=0, igb7=0, igb8=0)
refene2 = _ReferenceEnergy(igb2=10.356928, igb5=9.943308, igb7=7.020632, igb8=5.259028) # Implicit
refene2.solvent_energies(igb2=16.751825, igb5=15.661934, igb7=12.906876, igb8=15.024553) # Explicit
refene2.dielc2_energies()
refene2.dielc2.solvent_energies()
# Copying the reference energy to be printted on the old CPIN format
refene2_old = _ReferenceEnergy(igb2=10.356928, igb5=9.943308, igb7=7.020632, igb8=5.259028) # Implicit
refene2_old.solvent_energies(igb2=16.751825, igb5=15.661934, igb7=12.906876, igb8=15.024553) # Explicit
refene2_old.dielc2_energies()
refene2_old.dielc2.solvent_energies()
refene2_old.set_pKa(4.85, deprotonated=False)

PRN = TitratableResidue('PRN',
                        ['CA', 'HA1', 'HA2', 'CB', 'HB1', 'HB2', 'CG',
                        'O1', 'O2', 'H11', 'H12', 'H21', 'H22'], pka=4.85, typ="ph")

PRN.add_state(protcnt=0, refene=refene1, refene_old=refene1, pka_corr=0.0, # deprotonated
              charges=[-0.0508, -0.0173,
              -0.0173, 0.0026, -0.0425, -0.0425, 0.8054, -0.8188, -0.8188, 0.0,
              0.0, 0.0, 0.0])

PRN.add_state(protcnt=1, refene=refene2, refene_old=refene2_old, pka_corr=4.85, # protonated syn-O1
              charges=[-0.0181, 0.0256, 0.0256,
              -0.0284, 0.0430, 0.0430, 0.6801, -0.6511, -0.5838, 0.4641,
              0.0, 0.0, 0.0])

PRN.add_state(protcnt=1, refene=refene2, refene_old=refene2_old, pka_corr=4.85, # protonated anti-O1
              charges=[-0.0181, 0.0256, 0.0256,
              -0.0284, 0.0430, 0.0430, 0.6801, -0.6511, -0.5838, 0.0,
              0.4641, 0.0, 0.0])

PRN.add_state(protcnt=1, refene=refene2, refene_old=refene2_old, pka_corr=4.85, # protonated syn-O2
              charges=[-0.0181, 0.0256, 0.0256,
              -0.0284, 0.0430, 0.0430, 0.6801, -0.5838, -0.6511, 0.0,
              0.0, 0.4641, 0.0])

PRN.add_state(protcnt=1, refene=refene2, refene_old=refene2_old, pka_corr=4.85, # protonated anti-O2
              charges=[-0.0181, 0.0256, 0.0256,
              -0.0284, 0.0430, 0.0430, 0.6801, -0.5838, -0.6511, 0.0,
              0.0, 0.0, 0.4641])

PRN.check()


###########################     MY PROPIONIC ACID DEFINITION     ###########################

# For CA - HB2 I averaged between deprotonated and protonated state.
# For CG I folded the residual charge to get to integer total charge.
# For O1, O2, H11, H12, H21, H22 I used the same charges as in Amber's definition.

# Propionate (Pierilab definition)
refene1 = _ReferenceEnergy(igb1=0, igb2=0, igb5=0, igb7=0, igb8=0)
refene1.solvent_energies(igb1=0, igb2=0, igb5=0, igb7=0, igb8=0)
refene1.dielc2_energies(igb1=0, igb2=0, igb5=0, igb7=0, igb8=0)
refene1.dielc2.solvent_energies(igb1=0, igb2=0, igb5=0, igb7=0, igb8=0)
refene2 = _ReferenceEnergy(igb2=0, igb5=0, igb7=0, igb8=0) # Implicit
refene2.solvent_energies(igb2=0, igb5=0, igb7=0, igb8=0) # Explicit
refene2.dielc2_energies()
refene2.dielc2.solvent_energies()
# Copying the reference energy to be printted on the old CPIN format
refene2_old = _ReferenceEnergy(igb2=0, igb5=0, igb7=0, igb8=0) # Implicit
refene2_old.solvent_energies(igb2=0, igb5=0, igb7=0, igb8=0) # Explicit
refene2_old.dielc2_energies()
refene2_old.dielc2.solvent_energies()
refene2_old.set_pKa(4.85, deprotonated=False)

PAT = TitratableResidue('PAT',
                        ['CA', 'HA1', 'HA2', 'CB', 'HB1', 'HB2', 'CG',
                        'O1', 'O2', 'H11', 'H12', 'H21', 'H22'], pka=4.85, typ="ph")

PAT.add_state(protcnt=0, refene=refene1, refene_old=refene1, pka_corr=0.0, # deprotonated
              charges=[-0.03445, 0.00415,
              0.00415, -0.01290, 0.00025, 0.00025, 0.67615, -0.8188, -0.8188, 0.0,
              0.0, 0.0, 0.0])

PAT.add_state(protcnt=1, refene=refene2, refene_old=refene2_old, pka_corr=4.85, # protonated syn-O1
              charges=[-0.03445, 0.00415, 0.00415,
              -0.01290, 0.00025, 0.00025, 0.80935, -0.6511, -0.5838, 0.4641,
              0.0, 0.0, 0.0])

PAT.add_state(protcnt=1, refene=refene2, refene_old=refene2_old, pka_corr=4.85, # protonated anti-O1
              charges=[-0.03445, 0.00415, 0.00415,
              -0.01290, 0.00025, 0.00025, 0.80935, -0.6511, -0.5838, 0.0,
              0.4641, 0.0, 0.0])

PAT.add_state(protcnt=1, refene=refene2, refene_old=refene2_old, pka_corr=4.85, # protonated syn-O2
              charges=[-0.03445, 0.00415, 0.00415,
              -0.01290, 0.00025, 0.00025, 0.80935, -0.5838, -0.6511, 0.0,
              0.0, 0.4641, 0.0])

PAT.add_state(protcnt=1, refene=refene2, refene_old=refene2_old, pka_corr=4.85, # protonated anti-O2
              charges=[-0.03445, 0.00415, 0.00415,
              -0.01290, 0.00025, 0.00025, 0.80935, -0.5838, -0.6511, 0.0,
              0.0, 0.0, 0.4641])

PAT.check()
