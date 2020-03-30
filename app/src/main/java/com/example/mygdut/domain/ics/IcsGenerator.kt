package com.example.mygdut.domain.ics

import android.content.Context
import com.example.mygdut.data.ScheduleTImeGenerator
import com.example.mygdut.db.entity.Schedule
import com.example.mygdut.domain.SchoolCalendar
import net.fortuna.ical4j.model.DateTime
import net.fortuna.ical4j.model.component.VAlarm
import net.fortuna.ical4j.model.component.VEvent
import net.fortuna.ical4j.model.property.*
import net.fortuna.ical4j.util.FixedUidGenerator
import java.util.*


class IcsGenerator private constructor(
    private val data: List<Schedule>,
    private val timeGenerator: ScheduleTImeGenerator
) {
    //    private val sb = StringBuilder(
//        "BEGIN:VCALENDAR\n\n" +
//                "PRODID:-//Google Inc//Google Calendar 70.9054//EN\n" +
//                "VERSION:2.0\n" +
//                "CALSCALE:GREGORIAN\n" +
//                "METHOD:PUBLISH\n" +
//                "X-WR-CALNAME:${timeGenerator.getName()}\n" +
//                "X-WR-TIMEZONE:Asia/Shanghai\n" +
//                "X-WR-CALDESC:${timeGenerator.getName()}\n"
//    )
    private val calendar = net.fortuna.ical4j.model.Calendar().apply {
        properties.add(ProdId("-//Google Inc//Google Calendar 70.9054//EN"))
        properties.add(Version.VERSION_2_0)
        properties.add(CalScale.GREGORIAN)
    }
    private var timeClock: Int? = null
//    private val tz = kotlin.run {
//        val registry = TimeZoneRegistryFactory.getInstance().createRegistry()
//        val timezone = registry.getTimeZone("Asia/Shanghai")
//        timezone.vTimeZone
//    }


    private fun generateSchedule() {
        for (schedule in data) {
            val startList = timeGenerator.generateStartTime(schedule)
            val endList = timeGenerator.generateEndTime(schedule)
            for (i in startList.indices) {
                val event =
                    generateEvent(startList[i], endList[i], schedule.className, schedule.classRoom)
                timeClock?.let {
                    val alarm =
                        generateClock(schedule.className, schedule.classRoom, it, startList[i])
                    event.alarms.add(alarm)
                }
//                sb.append(event)
                calendar.components.add(event)
            }
        }
    }

    private fun generateEvent(start: Calendar, end: Calendar, name: String, place: String): VEvent {
        val startTime = DateTime(start.timeInMillis).apply { isUtc = true }
        val endTime = DateTime(end.timeInMillis).apply { isUtc = true }
        val event = VEvent(startTime, endTime, name)
        event.properties.add(Location(place))
        event.properties.add(FixedUidGenerator(HOST).generateUid())
//        event.properties.add(tz.timeZoneId)
//        val ssb = StringBuilder()
//        ssb.append("BEGIN:VEVENT\n")
//        ssb.append("DTSTART:${getTime(start)}\n")
//        ssb.append("DTEND:${getTime(end)}\n")
//        ssb.append("SUMMARY:$name\n")
//        ssb.append("LOCATION:$place\n")
//        ssb.append("END:VEVENT\n")
//        return ssb.toString()
        return event
    }

//    private fun getTime(calendar: Calendar): String {
//        val date = Date(calendar.timeInMillis)
//        val sdf1 = SimpleDateFormat("%Y%m%d", Locale.CHINESE)
//        val time1 = sdf1.format(date)
//        val sdf2 = SimpleDateFormat("%H%M%S", Locale.CHINESE)
//        val time2 = sdf2.format(date)
//        return "${time1}T${time2}Z"
//    }

    private fun generateClock(
        className: String,
        place: String,
        time: Int,
        startCal: Calendar
    ): VAlarm {
        startCal.add(Calendar.MINUTE, -time)
//                val timeCal = getTime(cal)
//                val ssb = StringBuilder("BEGIN:VALARM\n")
//                ssb.append("TRIGGER;VALUE=DATE-TIME:$timeCal\n")
//                ssb.append("ACTION:NONE\n")
//                ssb.append("END:VALARM\n")
//                sb.append(ssb.toString())
        val vAlarm = VAlarm(DateTime(startCal.timeInMillis))
        vAlarm.properties.add(Summary(className))
        vAlarm.properties.add(Action.DISPLAY)
        vAlarm.properties.add(Description("你有一门课程要上： $className, 课室: $place"))
        return vAlarm
    }

    private fun generate(): Ics {
//        sb.append("END:VCALENDAR")
//        return Ics(sb.toString(), timeGenerator.getTermName())
        calendar.validate()
        return Ics(calendar, timeGenerator.getTermName(),timeClock != null)
    }

    companion object {
        private const val HOST = "mygdut"
    }

    class Builder(data: List<Schedule>, schoolCalendar: SchoolCalendar, context: Context) {
        private val generator = IcsGenerator(data, ScheduleTImeGenerator(context, schoolCalendar))

        fun setTime(time: Int) {
            if (time >= 0) generator.timeClock = time
            else generator.timeClock = null
        }

        fun build(): Ics {
            generator.generateSchedule()
            return generator.generate()
        }
    }
}