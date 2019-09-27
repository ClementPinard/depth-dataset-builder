import piexif
from fractions import Fraction


def to_deg(value, loc):
    """convert decimal coordinates into degrees, munutes and seconds tuple

    Keyword arguments: value is float gps-value, loc is direction list ["S", "N"] or ["W", "E"]
    return: tuple like (25, 13, 48.343 ,'N')
    """
    if value < 0:
        loc_value = loc[0]
    elif value > 0:
        loc_value = loc[1]
    else:
        loc_value = ""
    abs_value = abs(value)
    deg = int(abs_value)
    t1 = (abs_value-deg)*60
    min = int(t1)
    sec = round((t1 - min) * 60, 5)
    return [deg, min, sec], loc_value


def to_dec(deg, min, sec, sign):
    result = deg[0] / deg[1]
    result += min[0] / (60 * min[1])
    result += sec[0] / (60 * 60 * sec[1])

    return sign * result


def change_to_rational(number):
    """convert a number to rantional

    Keyword arguments: number
    return: tuple like (1, 2), (numerator, denominator)
    """
    f = Fraction(str(number))
    return (f.numerator, f.denominator)


def set_gps_location(file_name, lat, lng, altitude):
    """Adds GPS position as EXIF metadata

    Keyword arguments:
    file_name -- image file
    lat -- latitude (as float)
    lng -- longitude (as float)
    altitude -- altitude (as float)

    """
    lat_deg, ref_lat = to_deg(lat, ["S", "N"])
    lng_deg, ref_lng = to_deg(lng, ["W", "E"])

    exif_lat = list(map(change_to_rational, lat_deg))
    exif_lng = list(map(change_to_rational, lng_deg))

    ref = 1 if altitude < 0 else 0

    gps_ifd = {
        piexif.GPSIFD.GPSVersionID: (2, 0, 0, 0),
        piexif.GPSIFD.GPSAltitudeRef: ref,
        piexif.GPSIFD.GPSAltitude: change_to_rational(abs(altitude)),
        piexif.GPSIFD.GPSLatitudeRef: ref_lat,
        piexif.GPSIFD.GPSLatitude: exif_lat,
        piexif.GPSIFD.GPSLongitudeRef: ref_lng,
        piexif.GPSIFD.GPSLongitude: exif_lng,
    }

    exif_dict = {"GPS": gps_ifd}
    exif_bytes = piexif.dump(exif_dict)
    piexif.insert(exif_bytes, file_name)


def get_gps_location(file_name):
    exif_dict = piexif.load(file_name)
    gps = exif_dict['GPS']
    if len(gps) == 0:
        return

    ref_lat = gps[piexif.GPSIFD.GPSLatitudeRef]
    ref_lng = gps[piexif.GPSIFD.GPSLongitudeRef]
    ref_alt = gps[piexif.GPSIFD.GPSAltitudeRef]

    exif_lat = gps[piexif.GPSIFD.GPSLatitude]
    exif_lng = gps[piexif.GPSIFD.GPSLongitude]
    exif_alt = gps[piexif.GPSIFD.GPSAltitude]

    lat = to_dec(*exif_lat, (1 if ref_lat == b'N' else -1))
    lng = to_dec(*exif_lng, (1 if ref_lng == b'E' else -1))

    alt = exif_alt[0] / exif_alt[1]
    if ref_alt == 1:
        alt *= -1

    return lat, lng, alt
