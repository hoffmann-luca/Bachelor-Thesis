# PowerShell: viele, leicht variierte Instanzen erzeugen
$out  = "\Test\out"
$meta = "\Test\meta"
$TOTAL = 500   # <- wie viele Instanzen erzeugen?

$Ns       = 600,800,1000,1200,1500
$taus     = 0.8,0.9,1.0,1.1,1.2,1.3
$tmx      = 5.0,10.0,16.0,18.0,20.0,22.5,25.0,30.0,50.0
$clusters = 2,3,4,5,6,8,12,15,18
$stds     = 0.03,0.05,0.06,0.08,0.10,0.12,0.15,0.18,0.2

for ($i=0; $i -lt $TOTAL; $i++) {
  $seed  = 45000 + $i
  $type  = @("uniform","clustered") | Get-Random
  $n     = $Ns | Get-Random
  $scoreMode = @("uniform","hotspots") | Get-Random

  $args = @(
    "--out", $out, "--meta", $meta,
    "--count", "1", "--prefix", "mix",
    "--n", $n, "--type", $type, "--scores", $scoreMode,
    "--seed", $seed
  )

  if ($type -eq "clustered") {
    $args += @("--clusters", ($clusters | Get-Random), "--cluster-std", ($stds | Get-Random))
  }

  if ($scoreMode -eq "hotspots") {
    $args += @("--score-min", (0,1 | Get-Random), "--score-max", (3,5,10 | Get-Random))
    $args += @("--hotspots", (1,2,3,4 | Get-Random), "--hotspot-bonus", (2,3,4,5,8 | Get-Random))
  }

  # Budget-Modus: 60% tau, 40% fixes tmax
  if ((Get-Random) % 100 -lt 60) {
    $args += @("--tau", ($taus | Get-Random))
  } else {
    $args += @("--tmax", ($tmx | Get-Random))
  }

  # ~30% der Fälle: Depot-Score auf 0
  if ((Get-Random) % 100 -lt 30) { $args += @("--depot-score-zero") }

  C:\Python3.6.3\python.exe gen_ttdp_instance_auto.py @args
}
Write-Host "`nFertig."
Read-Host "Enter zum Schließen"