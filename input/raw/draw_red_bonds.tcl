# draw_red_bonds.tcl
# Draw red cylinders between specific atom index pairs (0-based) in the *top* molecule.

# === settings you can tweak ===
set RADIUS 0.175
set COLOR  red
set MATERIAL Opaque   ;# or Transparent, Ghost, etc.

# === which bonds to draw (0-based indices) ===
set BONDS {
    {40 44}
    {40 35}
    {32 28}
    {28 23}
    {5  16}
    {16 20}
}

# === main ===
set mol [molinfo top]
if {$mol == -1} {
    error "No molecules loaded. Load your structure first, then 'source draw_red_bonds.tcl'."
}

# (Optional) clear previous custom drawings for this molecule:
# graphics $mol delete all

graphics $mol color $COLOR
graphics $mol material $MATERIAL

foreach pair $BONDS {
    lassign $pair i j

    # Build selections
    set sel1 [atomselect $mol "index $i"]
    set sel2 [atomselect $mol "index $j"]

    # Skip if an index is invalid for this mol
    if {[llength [$sel1 get index]] == 0 || [llength [$sel2 get index]] == 0} {
        puts "Warning: skipping bond {$i $j} (index not found in current molecule)."
        $sel1 delete
        $sel2 delete
        continue
    }

    # Coordinates for the current frame
    set p1 [lindex [$sel1 get {x y z}] 0]
    set p2 [lindex [$sel2 get {x y z}] 0]

    # Draw the cylinder
    graphics $mol cylinder $p1 $p2 radius $RADIUS

    # Clean up selections
    $sel1 delete
    $sel2 delete
}

puts "Done: drew [llength $BONDS] red cylinders in molecule $mol."
puts "Note: graphics are static per-frame and won't auto-update if you change frames."

