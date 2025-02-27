
# Version History
## 0.1.0
- Initial release

## 0.2.0
- Sand now slows down with a customizable probability (default 20%) instead of always.
- Bump environment version to v1 because the changes impact reproducibility with earlier versions.

## 0.3.0
- The x and y coordinates of observations are no longer swapped. This was the case for historical reasons but serves no use any more.
- Adds the option to use a sliding observation window of variable size.
- Adds the option to use the direction of the next subgoal as a additional observation.
- Bump environment version to v2 because the changes impact reproducibility with earlier versions.

## 0.3.1
- Fix bug that made it impossible to save a map as a file.

## 0.4.0
- Adds the option to have the start and goal of a map at custom or random positions.
- Bump environment version to v3 because the changes impact reproducibility with earlier versions.
- Update dependencies to allow for use with python versions ^3.10.

## 0.5.0
- Add the traffic light obstacle.
- Bump environment version to v4 because the changes impact reproducibility with earlier versions.