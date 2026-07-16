# Customizing your ATLAS installation

## Limiting the number of simultaneously submitted calculations

ATLAS includes a way of artificially limiting the number of submitted calculations.

In some supercomputer clusters there is a limit on the number of jobs that can be submitted at the same time, and going over this limit will result in the job not entering the queue.
AiiDA will reattempt to submit after the time given by `exponential_backoff_retry` is elapsed and for `task_maximum_attempts` attempts, and if the number of attempts is exceeded, the calculation will be paused, holding the loop.

This limit will be set by the supercomputer's scheduler, in the case of SLURM, it can be checked with `sacctmgr show qos XXXX` under the `MaxSubmitPU` column.

The artificial limit can be set by adding the `atl_calc_limit` property to the desired AiiDA Computer's metadata and it will be considered beforing a calculation in the library. The following code fragment shows how to do it for any Computer using its label. It can be executed line by line in the python interpreter launched via the `verdi shell` command or as a script.

```python
from aiida import orm


# Replace `computer_label_example` with the actual label of the AiiDA computer.
computer_name = 'computer_label_example'

# Load the computer
computer = orm.load_computer(computer_name)

# Set the custom property
computer.set_property(name='atl_calc_limit', value=350)
```

## Changing the default image viewer during SSH usage

ATLAS includes a custom `ImagePNGData` class that wraps AiiDA's `SinglefileData`. The custom image class includes some utility methods to display images automatically, since plots are automatically generated during active learning loops and database generation.

To visualize an image, get the PK/UUID of a ImagePNGData node and run:

```bash
verdi data atl.img-png show <PK/UUID>
```

This will display the image using the default program for the `image/png` mimetype. You can check the default by running:

```bash
xdg-mime query default image/png
```

In Ubuntu distributions running gnome, the usual program is EOG (`org.gnome.eog.desktop`) although this can be changed by the user. However, when using `xdg-open` through SSH or a session where certain environment variables are not available (`$DISPLAY`, ...), xdg-open drops into a generic fallback mode and scans standard CLI lists (usually `/etc/mailcap`) for programs that don't require a GUI, such as `cacaview`.

You can create a user-level override that sets any CLI-useable program as the default for images, but apply a test so it only executes if there is no graphical display by editing `~/.mailcap`

```bash
image/png; <PROGRAM NAME> %s; test=test -z "$DISPLAY" -a -z "$WAYLAND_DISPLAY"
```

For visualization of the plots in high quality through SSH our advice is to install the `wezterm` terminal emulator in both local and remote nodes, and use `wezterm` as the terminal emulator to connect to the remote node through SSH. This program provides a `wezterm imgcat` command that displays images without quality loss over SSH:

```bash
image/png; wezterm imgcat %s; test=test -z "$DISPLAY" -a -z "$WAYLAND_DISPLAY"
```
