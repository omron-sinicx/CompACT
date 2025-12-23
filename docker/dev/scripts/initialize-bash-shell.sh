#!/bin/bash

################################################################################

# Link the default shell 'sh' to Bash.
alias sh='/bin/bash'

################################################################################

# Configure the terminal.

# Disable flow control. If enabled, inputting 'ctrl+s' locks the terminal until inputting 'ctrl+q'.
stty -ixon

################################################################################

# Configure 'umask' for giving read/write/execute permission to group members.
umask 0002

################################################################################

# Add the Catkin workspaces to the 'ROS_PACKAGE_PATH'.
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:/root/compact_env/catkin_ws/src/
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:/root/compact_env/underlay_ws/src/

################################################################################

# Define Bash functions to conveniently execute the helper scripts in the current shell process.

function osx-repair-git-paths () {
  # Store the current directory and execute scripts in the current shell process.
  pushd .
  source /root/compact_env/docker/dev/scripts/repair-git-paths.sh
  popd
}

function osx-fix-permission-issues () {
  # Store the current directory and execute scripts in the current shell process.
  pushd .
  source /root/compact_env/docker/dev/scripts/repair-git-paths.sh
  source /root/compact_env/docker/dev/scripts/fix-permission-issues.sh
  popd
}

# Move to the working directory.
cd /root/compact_env/
