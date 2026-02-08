## Hotfix for: tkcvs/noble,noble,now 8.2.3-1.2
[Bug 2139062](https://bugs.launchpad.net/ubuntu/+source/tkcvs/+bug/2139062)<br>
### Fix description
Replace line 3717:
```tcl
    set factor [expr {$mapheight / $lines}]
```
By:
```tcl
    set factor [expr {$lines > 0 ? $mapheight / double($lines) : 1.0}]
```

One line command:
```tcsh
sudo sed -i.bak 's@set factor \[expr {$mapheight / $lines}\]@set factor [expr {$lines > 0 ? $mapheight / double($lines) : 1.0}]@' /usr/bin/tkdiff
```
